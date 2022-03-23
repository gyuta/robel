from robel.dkitty.distral.base import Base
import numpy as np
from typing import Dict, Optional, Sequence, Tuple, Union
from robel.components.tracking import TrackerComponentBuilder, TrackerState
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
