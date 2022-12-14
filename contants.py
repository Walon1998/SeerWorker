import numpy as np

HALF_LIFE_SECONDS = 10
ACIONS_PER_SECOND = 15
GAMMA = np.exp(np.log(0.5) / (ACIONS_PER_SECOND * HALF_LIFE_SECONDS))
GAE_LAMBDA = 0.95
N_STEPS = 512
PAST_MODELS = 0.2
