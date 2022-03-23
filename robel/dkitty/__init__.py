# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gym environment registration for DKitty environments."""

from robel.utils.registration import register

#===============================================================================
# Stand tasks
#===============================================================================

# Default number of steps per episode.
_STAND_EPISODE_LEN = 80  # 80*40*2.5ms = 8s

register(
    env_id='DKittyStandFixed-v0',
    class_path='robel.dkitty.stand:DKittyStandFixed',
    max_episode_steps=_STAND_EPISODE_LEN)

register(
    env_id='DKittyStandRandom-v0',
    class_path='robel.dkitty.stand:DKittyStandRandom',
    max_episode_steps=_STAND_EPISODE_LEN)

register(
    env_id='DKittyStandRandomDynamics-v0',
    class_path='robel.dkitty.stand:DKittyStandRandomDynamics',
    max_episode_steps=_STAND_EPISODE_LEN)

#===============================================================================
# Orient tasks
#===============================================================================

# Default number of steps per episode.
_ORIENT_EPISODE_LEN = 80  # 80*40*2.5ms = 8s

register(
    env_id='DKittyOrientFixed-v0',
    class_path='robel.dkitty.orient:DKittyOrientFixed',
    max_episode_steps=_ORIENT_EPISODE_LEN)

register(
    env_id='DKittyOrientRandom-v0',
    class_path='robel.dkitty.orient:DKittyOrientRandom',
    max_episode_steps=_ORIENT_EPISODE_LEN)

register(
    env_id='DKittyOrientRandomDynamics-v0',
    class_path='robel.dkitty.orient:DKittyOrientRandomDynamics',
    max_episode_steps=_ORIENT_EPISODE_LEN)

#===============================================================================
# Walk tasks
#===============================================================================

# Default number of steps per episode.
_WALK_EPISODE_LEN = 160  # 160*40*2.5ms = 16s

register(
    env_id='DKittyWalkFixed-v0',
    class_path='robel.dkitty.walk:DKittyWalkFixed',
    max_episode_steps=_WALK_EPISODE_LEN)

register(
    env_id='DKittyWalkRandom-v0',
    class_path='robel.dkitty.walk:DKittyWalkRandom',
    max_episode_steps=_WALK_EPISODE_LEN)

register(
    env_id='DKittyWalkRandomDynamics-v0',
    class_path='robel.dkitty.walk:DKittyWalkRandomDynamics',
    max_episode_steps=_WALK_EPISODE_LEN)



# 追加タスク

LEN = 160

register(
    env_id="walk-v0",
    class_path="robel.dkitty.distral.dwalk:Walk",
    max_episode_steps=LEN
)
register(
    env_id="orient-v0",
    class_path="robel.dkitty.distral.dorient:Orient",
    max_episode_steps=LEN
)
register(
    env_id="walkrandom-v0",
    class_path="robel.dkitty.distral.dwalk:WalkRandom",
    max_episode_steps=LEN
)
register(
    env_id='walkrandom-v1',
    class_path='robel.dkitty.walk:DKittyWalkRandom',
    max_episode_steps=_WALK_EPISODE_LEN)

register(
    env_id="turnwalk-v0",
    class_path="robel.dkitty.distral.dwalk:TurnWalk",
    max_episode_steps=LEN
)


register(
    env_id="orient_random-v0",
    class_path="robel.dkitty.distral.dorient:OrientRandom",
    max_episode_steps=LEN
)
register(
    env_id="walk_random-v0",
    class_path="robel.dkitty.distral.dwalk:WalkRandomDistance",
    max_episode_steps=LEN
)

register(
    env_id="leftorient_random-v0",
    class_path="robel.dkitty.distral.dorient:LeftOrientRandom",
    max_episode_steps=80
)
register(
    env_id="leftorient_random-v1",
    class_path="robel.dkitty.distral.dorient:LeftOrientRandom2",
    max_episode_steps=80
)

register(
    env_id="orient_coordrandom-v0",
    class_path="robel.dkitty.distral.dorient:OrientCoordRandom",
    max_episode_steps=80
)

register(
    env_id="turnwalk_coordrandom-v0",
    class_path="robel.dkitty.distral.dwalk:TurnWalkCoordRandom",
    max_episode_steps=LEN
)

register(
    env_id="turnwalk_cr-rd-v0",
    class_path="robel.dkitty.distral.dwalk:TurnWalkRandomDynamics",
    max_episode_steps=LEN
)

register(
    env_id="dwalkrandom-v0",
    class_path="robel.dkitty.distral.dwalk:DWalkRandom",
    max_episode_steps=LEN
)

register(
    env_id="dwalk_rd-v0",
    class_path="robel.dkitty.distral.dwalk:DWalkRandomDynamics",
    max_episode_steps=LEN
)

register(
    env_id="orient_cr-rd-v0",
    class_path="robel.dkitty.distral.dorient:OrientCoordRandomDynamics",
    max_episode_steps=80
)