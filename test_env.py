import gym
import env_resource.envs.robotics
import numpy as np

if __name__ == '__main__':
    env = gym.make('DoubleFetchStackEnv-v0')
    # env = gym.make('PushEnv-v3')
    obs = env.reset()
    for _ in range(10000):
        env.step(2 * np.random.rand(8) - 1.)
        env.render()
