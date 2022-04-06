from robel.dkitty.distral.base import Base
import numpy as np
from typing import Dict, Optional, Sequence, Tuple, Union
from robel.components.tracking import TrackerComponentBuilder, TrackerState


from robel.components.tracking import TrackerComponentBuilder, TrackerState
from robel.dkitty.base_env import BaseDKittyUprightEnv
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.math_utils import calculate_cosine
from robel.utils.resources import get_asset_path
class Walk(Base):
    """ walk task

    Baseがfixedwalkを継承しているので特に修正を加えなくてもよい
    """
    pass

class WalkRandom(Base):
    def __init__(
            self,
            *args,
            target_distance_range: Tuple[float, float] = (1.0, 2.0),
            # +/- 60deg
            target_angle_range: Tuple[float, float] = (-np.pi / 3, np.pi / 3),
            **kwargs):
        """Initializes the environment.

        Args:
            target_distance_range: The range in which to sample the target
                distance.
            target_angle_range: The range in which to sample the angle between
                the initial D'Kitty heading and the target.
        """
        super().__init__(*args, **kwargs)
        self._target_distance_range = target_distance_range
        self._target_angle_range = target_angle_range

    def _reset(self):
        """Resets the environment."""
        target_dist = self.np_random.uniform(*self._target_distance_range)
        # Offset the angle by 90deg since D'Kitty looks towards +y-axis.
        target_theta = np.pi / 2 + self.np_random.uniform(
            *self._target_angle_range)
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta), 0
        ])
        super()._reset()
        print(target_dist, target_theta)

class TurnWalk(Walk):
    """ ひっくり返って歩く
    """
    def _reset(self):
        """Resets the environment."""

        self._reset_dkitty_standing()
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot_euler=np.array([0,0,np.pi])),
            'target': TrackerState(pos=np.array([0,2,0])),
            'heading': TrackerState(pos=np.array([0,2,0])),
        })

class TurnWalkCoordRandom(Walk):
    """ OrientCoordRandom に対応したタスク
    """
    def _reset(self):
        """Resets the environment."""
        p = (np.random.random(3)-0.5)*2
        p[2] = 0

        self._reset_dkitty_standing()
        self.tracker.set_state({
            'torso': TrackerState(pos=p, rot_euler=np.array([0,0,np.pi])),
            'target': TrackerState(pos=np.array([0,2,0])),
            'heading': TrackerState(pos=np.array([0,2,0])),
        })

@configurable(pickleable=True)
class TurnWalkRandomDynamics(TurnWalkCoordRandom):
    """Walk straight towards a random location."""

    def __init__(self,
                 *args,
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2.8, 3.2),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, 0.05),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()


class WalkRandomDistance(Walk):
    def _reset(self):
        """Resets the environment."""
        self._reset_dkitty_standing()

        # If no heading is provided, head towards the target.
        dist = np.random.random() + 1.5
        target_pos = np.array([0,dist,0])

        # Set the tracker locations.
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot=np.identity(3)),
            'target': TrackerState(pos=target_pos),
            'heading': TrackerState(pos=target_pos),
        })

class DWalkRandom(Walk):
    """ ゴール固定でDKittyの位置をランダム
    """
    
    def _reset(self):
        """Resets the environment."""
        p = (np.random.random(3)-0.5)*2
        p[2] = 0

        self._reset_dkitty_standing()
        self.tracker.set_state({
            'torso': TrackerState(pos=p, rot_euler=np.array([0,0,0])),
            'target': TrackerState(pos=np.array([0,2,0])),
            'heading': TrackerState(pos=np.array([0,2,0])),
        })

@configurable(pickleable=True)
class DWalkRandomDynamics(DWalkRandom):
    """Walk straight towards a random location."""

    def __init__(self,
                 *args,
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2.8, 3.2),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, 0.05),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()