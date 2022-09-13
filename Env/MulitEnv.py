import time

import gym
import numpy as np

from Env.AsyncEnv import AsyncEnv


class MultiEnv(gym.Env):

    def __init__(self, num_instances):
        super(MultiEnv, self).__init__()

        self.num_instances = num_instances

        self.obs_shape = (self.num_instances * 2, 159)
        self.reward_shape = (self.num_instances * 2)

        self.instances = []

        for i in range(num_instances):
            self.instances.append(AsyncEnv())

        self.observation_space = self.instances[0].observation_space
        self.action_space = self.instances[0].action_space

        self.num_envs = self.num_instances * 2
        self.is_vector_env = True

    def reset(self):
        obs = np.empty(self.obs_shape, dtype=np.float32)
        counter = 0
        for i in self.instances:
            obs[counter: counter + 2, :] = i.reset()
            counter += 2
        return obs

    def step(self, action):
        obs_array = np.empty(self.obs_shape, dtype=np.float32)
        reward_array = np.empty(self.reward_shape, dtype=np.float32)
        done_array = np.empty(self.reward_shape, dtype=np.bool)

        counter = 0
        for i in self.instances:
            obs, rew, done, info = i.step(action[counter: counter + 2, :])
            obs_array[counter: counter + 2, :] = obs
            reward_array[counter: counter + 2] = rew
            done_array[counter: counter + 2] = done
            counter += 2

        return obs_array, reward_array, done_array, None

    def render(self, mode="human"):
        pass
