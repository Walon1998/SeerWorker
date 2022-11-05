from typing import Any

import gym.spaces
import numpy as np
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState


class SeerActionV2(ActionParser):
    def __init__(self, flip_bins=3, throttle_bins=3, roll_bins=3):
        super(ActionParser).__init__()
        assert flip_bins % 2 == 1, "n_bins must be an odd number"
        assert throttle_bins % 2 == 1, "n_bins must be an odd number"
        assert roll_bins % 2 == 1, "throttle_bins must be an odd number"

        self.flip_bins = flip_bins
        self.throttle_bins = throttle_bins
        self.roll_bins = roll_bins

        assert self.flip_bins == 3
        assert self.throttle_bins == 3
        assert self.roll_bins == 3

    def get_action_space(self) -> gym.spaces.Space:
        # throttle, steer/yaw, pitch, roll, jump, boost, handbrake
        return gym.spaces.MultiDiscrete([self.throttle_bins] + [self.flip_bins] * 2 + [self.roll_bins] + [2] * 3)

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        actions = actions.reshape((-1, 7)).astype(dtype=np.float32)
        steer_yaw = actions[:, 1] - 1.0  # [0,1,2] -> [-1,0,1]

        parsed = np.empty((actions.shape[0], 8), dtype=np.float32)
        parsed[:, 0] = actions[:, 0] - 1.0  # throttle [0,1,2] -> [-1,0,1]
        parsed[:, 1] = steer_yaw  # steer
        parsed[:, 2] = actions[:, 2] - 1.0  # pitch [0,1,2] -> [-1,0,1]
        parsed[:, 3] = steer_yaw  # yaw
        parsed[:, 4] = actions[:, 3] - 1.0  # roll [0,1,2] -> [-1,0,1]
        parsed[:, 5] = actions[:, 4]  # jump
        parsed[:, 6] = actions[:, 5]  # boost
        parsed[:, 7] = actions[:, 6]  # handbrake

        return parsed
