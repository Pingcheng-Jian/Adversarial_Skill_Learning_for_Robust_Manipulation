import sys

sys.path.append('../')
from gym.envs.registration import register
import gym


register(
    id='DoublePick-v1',
    entry_point='env_resource.envs.robotics.double_fetch_stack:DoubleFetchStackEnv_v1',
    kwargs={'reward_type': 'sparse', 'random_gripper': True, 'random_box': True, 'random_ratio': 1.0,
            'n_object': 1, 'n_robot': 2, '_all': True}
)
