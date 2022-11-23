import time

import gym
import numpy as np

from Env.AsyncEnv import AsyncEnv


def get_obs_size(team_size):
    sizes = [103, 147, 191]
    return sizes[team_size - 1]


class MultiEnv(gym.Env):

    def __init__(self, num_instances, team_size, force_paging):
        super(MultiEnv, self).__init__()

        self.num_instances = num_instances
        self.team_size = team_size

        assert self.num_instances > 0

        self.obs_shape = (self.num_instances * 2 * team_size, get_obs_size(self.team_size))
        self.reward_shape = (self.num_instances * 2 * team_size)

        self.instances = []

        for i in range(num_instances):
            self.instances.append(AsyncEnv(team_size, force_paging))

        self.observation_space = self.instances[0].observation_space
        self.action_space = self.instances[0].action_space

        self.num_envs = self.num_instances * 2 * team_size
        self.is_vector_env = True

    def reset(self):
        obs = np.empty(self.obs_shape, dtype=np.float32)

        for i in self.instances:
            i.reset_put()

        counter = 0
        for i in self.instances:
            obs[counter: counter + 2 * self.team_size, :] = i.reset_get()
            counter += 2 * self.team_size
        return obs

    def step(self, action):

        counter = 0
        for i in self.instances:
            i.step_put(action[counter: counter + 2 * self.team_size, :])
            counter += 2 * self.team_size

        obs_array = np.empty(self.obs_shape, dtype=np.float32)
        reward_array = np.empty(self.reward_shape, dtype=np.float32)
        done_array = np.empty(self.reward_shape, dtype=np.bool)

        counter = 0
        for i in self.instances:
            obs, rew, done, info = i.step_get()
            obs_array[counter: counter + 2 * self.team_size, :] = obs
            reward_array[counter: counter + 2 * self.team_size] = rew
            done_array[counter: counter + 2 * self.team_size] = done
            counter += 2 * self.team_size

        return obs_array, reward_array, done_array, None

    def render(self, mode="human"):
        pass
