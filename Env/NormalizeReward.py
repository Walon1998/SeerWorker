import numpy as np
import gym
from numba import int32, float32, jit, float64
from numba.experimental import jitclass

spec = [
    ('mean', float64),
    ('var', float64),
    ('count', float64),
]


@jitclass(spec)
class RunningMeanStd:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        self.update_from_moments(batch_mean, batch_var)

    def update_from_moments(self, batch_mean, batch_var):
        self.mean, self.var,  = update_mean_var_count_from_moments(
            self.mean, self.var,  batch_mean, batch_var
        )



@jit(fastmath=True)
def update_mean_var_count_from_moments(
        mean, var,  batch_mean, batch_var
):
    MOMENTUM = 0.001
    new_mean = (1 - MOMENTUM) * mean + MOMENTUM * batch_mean
    new_var = (1 - MOMENTUM) * var + MOMENTUM * batch_var
    return new_mean, new_var


class NormalizeReward(gym.core.Wrapper):
    def __init__(
            self,
            env,
            mean,
            var,
            gamma=0.99,
            epsilon=1e-8,
    ):
        super(NormalizeReward, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(mean, var)
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        self.returns[dones] = 0.0
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, dones, infos

    def normalize(self, rews):
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)

    def get_monitor_data(self):
        return self.env.get_monitor_data()
