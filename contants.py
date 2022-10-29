import numpy as np
from shared_memory_dict import SharedMemoryDict


class Annealer:
    def __init__(self, val_old, val_new, start, end):
        self.val_old = val_old
        self.val_new = val_new
        self.start = start
        self.end = end
        self.smd_config = SharedMemoryDict(name='shared_memory_dict', size=1024)
        assert end >= start
        self.steps = end - start

    def get(self):

        if self.smd_config["counter"] < self.start:
            return self.val_old

        if self.smd_config["counter"] >= self.end:
            return self.val_new

        frac = (self.smd_config["counter"] - self.start) / self.steps

        val = frac * self.val_new + (1 - frac) * self.val_old

        return val


HALF_LIFE_SECONDS_OLD = 10
HALF_LIFE_SECONDS_NEW = 20
ACIONS_PER_SECOND = 15
GAMMA_OLD = np.exp(np.log(0.5) / (ACIONS_PER_SECOND * HALF_LIFE_SECONDS_OLD))
GAMMA_NEW = np.exp(np.log(0.5) / (ACIONS_PER_SECOND * HALF_LIFE_SECONDS_NEW))
GAMMA_Annealer = Annealer(GAMMA_OLD, GAMMA_NEW, 30_000, 32_500)
LSTM_UNROLL_LENGTH = 16
GAE_LAMBDA = 0.95
N_STEPS = 1024
PAST_MODELS = 0.2
