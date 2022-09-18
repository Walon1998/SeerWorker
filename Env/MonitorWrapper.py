import numpy as np
from gym import Wrapper, Env


class MonitorWrapper(Wrapper):
    def __init__(self, env):
        super(MonitorWrapper, self).__init__(env)

        self.episode_returns = None
        self.buf = []

    def reset(self):
        obs = self.env.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.buf = []
        return obs

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)

        self.episode_returns += rewards

        for i in range(dones.shape[0]):
            if dones[i]:
                self.buf.append(self.episode_returns[i])
                self.episode_returns[i] = 0

        return obs, rewards, dones, infos

    def get_mean_returns(self):

        if len(self.buf) == 0:
            return 0.0

        reward_mean = np.mean(self.buf)

        self.buf = []

        return reward_mean
