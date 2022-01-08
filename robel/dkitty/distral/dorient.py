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
        self._initial_angle = np.pi
        # self._target_angle = np.pi

        self._reset_dkitty_standing()
        # Set the initial target position.
        self.tracker.set_state({
            'torso': TrackerState(
                pos=np.zeros(3),
                rot_euler=np.array([0, 0, self._initial_angle])),
            # 'target': TrackerState(
            #     pos=np.array([
            #         # The D'Kitty is offset to face the y-axis.
            #         np.cos(self._target_angle + np.pi / 2),
            #         np.sin(self._target_angle + np.pi / 2),
            #         0,
            #     ])),
        })