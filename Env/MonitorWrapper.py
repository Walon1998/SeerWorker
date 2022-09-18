import numpy as np
from gym import Wrapper, Env


class MonitorWrapper(Wrapper):
    def __init__(self, env):
        super(MonitorWrapper, self).__init__(env)

        self.episode_returns = None
        self.episode_lengths = None
        self.buf = []

    def reset(self):
        obs = self.env.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.buf = []
        return obs

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)

        self.episode_returns += rewards
        self.episode_lengths += 1

        for i in range(dones.shape[0]):
            if dones[i]:
                data = self.episode_returns[i], self.episode_lengths[i]
                self.buf.append(data)
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0

        return obs, rewards, dones, infos

    def get_monitor_data(self):
        data = np.array([i[0] for i in self.buf], dtype=np.float32), np.array([i[1] for i in self.buf], dtype=np.float32)
        self.buf = []
        return data
