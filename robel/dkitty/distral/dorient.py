import numpy as np
import collections

from robel.dkitty.distral.base import Base
from robel.components.tracking import TrackerComponentBuilder, TrackerState

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
