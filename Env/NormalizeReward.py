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
class RunningMeanStd_Old:
    def __init__(self, mean, var, epsilon=1e-4):
        self.mean = mean
        self.var = var
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments_old(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


@jit(fastmath=True)
def update_mean_var_count_from_moments_old(
        mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


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
        self.return_rms = RunningMeanStd_Old(mean, var)
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
