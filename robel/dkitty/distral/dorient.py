import numpy as np
import collections

from robel.dkitty.distral.base import Base
from robel.components.tracking import TrackerComponentBuilder, TrackerState

from robel.components.tracking import TrackerComponentBuilder, TrackerState
from robel.dkitty.base_env import BaseDKittyUprightEnv
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.math_utils import calculate_cosine
from robel.utils.resources import get_asset_path
from typing import Dict, Optional, Sequence, Tuple, Union

class Orient(Base):
    def get_reward_dict(self, action, obs_dict):
        heading = obs_dict['heading']
        angle_error = np.arccos(heading)
        upright = obs_dict[self._upright_obs_key]
        center_dist = np.linalg.norm(obs_dict['root_pos'][:2], axis=1)

        reward_dict = collections.OrderedDict((
            # Add reward terms for being upright.
            *self._get_upright_rewards(obs_dict).items(),
            # Reward for closeness to desired facing direction.
            ('alignment_error_cost', -4 * angle_error),
            # Reward for closeness to center; i.e. being stationary.
            ('center_distance_cost', -4 * center_dist),
            # Bonus when mean error < 15deg or upright within 15deg.
            ('bonus_small', 5 * ((angle_error < 0.26) + (upright > 0.96))),
            # Bonus when error < 5deg and upright within 15deg.
            ('bonus_big', 10 * ((angle_error < 0.087) * (upright > 0.96))),
        ))
        return reward_dict
    
    def _reset(self):
        """Resets the environment."""

        self._reset_dkitty_standing()
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot_euler=np.array([0,0,np.pi])),
            'target': TrackerState(pos=np.array([0,2,0])),
            'heading': TrackerState(pos=np.array([0,2,0])),
        })

class OrientCoordRandom(Orient):
    """ 角度は固定で初期胴体の座標だけをランダムにしたオリエント
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
class OrientCoordRandomDynamics(OrientCoordRandom):
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

class OrientRandom(Orient):
    def _reset(self):
        """Resets the environment."""
        ang = np.random.random()*np.pi/3 + np.pi - np.pi/6
        self._reset_dkitty_standing()
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot_euler=np.array([0,0,ang])),
            'target': TrackerState(pos=np.array([0,2,0])),
            'heading': TrackerState(pos=np.array([0,2,0])),
        })


class LeftOrientRandom(Orient):
    def _reset(self):
        """Resets the environment."""
        ang = np.random.random()*np.pi/3 + np.pi - np.pi/3
        self._reset_dkitty_standing()
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot_euler=np.array([0,0,ang])),
            'target': TrackerState(pos=np.array([0,2,0])),
            'heading': TrackerState(pos=np.array([0,2,0])),
        })


class LeftOrientRandom2(Orient):
    def _reset(self):
        """Resets the environment."""
        ang = np.random.random()*np.pi/6 + np.pi - np.pi/6
        self._reset_dkitty_standing()
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot_euler=np.array([0,0,ang])),
            'target': TrackerState(pos=np.array([0,2,0])),
            'heading': TrackerState(pos=np.array([0,2,0])),
        })
