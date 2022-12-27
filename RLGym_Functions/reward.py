import math
from typing import Union

import numpy as np
import pandas as pd
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import BALL_RADIUS, BALL_MAX_SPEED, CAR_MAX_SPEED, CEILING_Z
from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards import SaveBoostReward, LiuDistancePlayerToBallReward, LiuDistanceBallToGoalReward, FaceBallReward, RewardIfClosestToBall, \
    AlignBallGoal, ConstantReward, RewardIfTouchedLast, RewardIfBehindBall, VelocityPlayerToBallReward, VelocityReward, EventReward
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
from shared_memory_dict import SharedMemoryDict
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM
import seaborn as sns
from matplotlib import pyplot as plt

from contants import GAMMA


class AirReward(RewardFunction):
    def __init__(self):
        super(AirReward, self).__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.on_ground:
            return 0.0
        else:
            return player.car_data.position[2] / CEILING_Z


class TouchedLast(RewardFunction):
    def __init__(self):
        super(TouchedLast, self).__init__()
        self.last_touch = None

    def reset(self, initial_state: GameState):
        self.last_touch = None

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            self.last_touch = player.car_id

        return player.car_id == self.last_touch


class DribbleReward(RewardFunction):
    def __init__(self):
        super(RewardFunction, self).__init__()
        self.potential = {}
        self.SUBTRACT = 0.05

    def reset(self, initial_state: GameState):
        self.potential = {}

        for p in initial_state.players:
            self.potential[p.car_id] = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        if player.ball_touched:
            self.potential[player.car_id] = 1.0
        else:
            self.potential[player.car_id] = max(self.potential[player.car_id] - self.SUBTRACT, 0)

        return self.potential[player.car_id]


class ForwardVelocity(RewardFunction):
    def __init__(self):
        super(ForwardVelocity, self).__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel_norm = player.car_data.linear_velocity / CAR_MAX_SPEED
        return float(np.dot(player.car_data.forward(), vel_norm))


class GoalScoredReward(RewardFunction):
    def __init__(self, ball_speed_bonus=0.1):
        super(GoalScoredReward, self).__init__()
        self.ball_speed_bonus = ball_speed_bonus
        self.prev_state = None
        self.rewards = {}

    def reset(self, initial_state: GameState):
        self.prev_state = initial_state
        self.rewards = {}
        for player in initial_state.players:
            self.rewards[player.car_id] = 0

    def pre_step(self, state: GameState):

        blue_scored = state.blue_score > self.prev_state.blue_score
        orange_scored = state.orange_score > self.prev_state.orange_score

        bonus = self.ball_speed_bonus * np.linalg.norm(self.prev_state.ball.linear_velocity) / BALL_MAX_SPEED

        for player in state.players:

            if blue_scored and player.team_num == BLUE_TEAM:
                self.rewards[player.car_id] = 1 + bonus

            elif orange_scored and player.team_num == ORANGE_TEAM:
                self.rewards[player.car_id] = 1 + bonus
            else:
                self.rewards[player.car_id] = 0

        self.prev_state = state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.rewards[player.car_id]


class DemoReward(RewardFunction):
    def __init__(self):
        super(DemoReward, self).__init__()
        # Need to keep track of last registered value to detect changes
        self.last_registered_values = {}

    def reset(self, initial_state: GameState):
        self.last_registered_values = {}
        for player in initial_state.players:
            self.last_registered_values[player.car_id] = player.match_demolishes

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        reward = 0
        if player.match_demolishes > self.last_registered_values[player.car_id]:
            reward = 1
        self.last_registered_values[player.car_id] = player.match_demolishes
        return reward


class SeerTouchBallReward(RewardFunction):
    def __init__(self, aerial_weight=0.0, decay=0.95, min_value=0.1, add_epsilon=0.013):
        super(SeerTouchBallReward, self).__init__()
        self.aerial_weight = aerial_weight
        self.decay = decay
        self.min_value = min_value
        self.add_epsilon = add_epsilon
        self.players = {}
        assert decay < 1.0

    def reset(self, initial_state: GameState):
        self.players.clear()

        for p in initial_state.players:
            self.players[p.car_id] = 1.0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        decay_factor = self.players.get(player.car_id)

        if player.ball_touched:
            self.players[player.car_id] = max(decay_factor * self.decay, self.min_value)
            r = ((state.ball.position[2] + BALL_RADIUS) / (2 * BALL_RADIUS)) ** self.aerial_weight
            return decay_factor * r

        else:
            self.players[player.car_id] = min(1.0, decay_factor + self.add_epsilon)
            return 0.0


class SeerRewardV2(RewardFunction):
    def __init__(self):
        super(RewardFunction, self).__init__()

        self.rewards = [
            GoalScoredReward(0.1),
            DemoReward(),
        ]

        self.rewards_weights = np.array([
            1.0,  # Goal Scored, Sparse, {0,1}
            0.1,  # Demo, Sparse, {0,1}
        ], dtype=np.float32)

        self.potentials = [
            LiuDistancePlayerToBallReward(),
            LiuDistanceBallToGoalReward(False),
            FaceBallReward(),
            AlignBallGoal(0.5, 0.5),
            RewardIfClosestToBall(ConstantReward(), False),
            TouchedLast(),
            RewardIfBehindBall(ConstantReward()),
            VelocityPlayerToBallReward(False),
            VelocityReward(False),
            ForwardVelocity(),
            SaveBoostReward(),
            DribbleReward()
        ]

        self.potential_weights = np.array([
            1.0,  # Distance Ball Player, Cont., [0,1]
            1.0,  # Distance Ball Goal, Cont., [0,1]
            0.25,  # Face Ball, Cont., [-1,1]
            1.0,  # Align ball goal, cont, [-1,1]
            0.25,  # Closest to ball, cont, {0,1}
            0.5,  # Touched Last, cont, {0,1}
            0.25,  # Behind Ball, cont, {0,1},
            0.25,  # velocity to ball, cont, [-1,1]
            0.25,  # velocity, cont, [0,1]
            0.1,  # forward velocity, cont, [-1,1]
            1.0,  # Save Boost, cont, [0,1]
            1.0,  # DribbleReward [0,1]
        ], dtype=np.float32)

        self.theta_last = 0

        assert len(self.rewards_weights) == len(self.rewards)
        assert len(self.potential_weights) == len(self.potentials)

    def reset(self, initial_state: GameState):

        self.theta_last = 0

        for r in self.rewards:
            r.reset(initial_state)

        for r in self.potentials:
            r.reset(initial_state)

    def pre_step(self, state: GameState):

        for r in self.potentials:
            r.pre_step(state)

        for r in self.rewards:
            r.pre_step(state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        rewards = [r.get_reward(player, state, previous_action) for r in self.rewards]
        potentials = [r.get_reward(player, state, previous_action) for r in self.potentials]

        R = np.dot(rewards, self.rewards_weights)

        theta_now = np.dot(potentials, self.potential_weights)

        F = GAMMA * theta_now - self.theta_last

        self.theta_last = theta_now

        return R + F

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)


class DistributeRewardsV2(RewardFunction):
    """
    Inspired by OpenAI's Dota bot (OpenAI Five).
    Modifies rewards using the formula (1-team_spirit) * own_reward + team_spirit * avg_team_reward - avg_opp_reward
        For instance, in a 3v3 where scoring a goal gives 100 reward with team_spirit 0.3 / 0.6 / 0.9:
            - Goal scorer gets 80 / 60 / 40
            - Teammates get 10 / 20 / 30 each
            - Opponents get -33.3 each
    Note that this will bring mean reward close to zero, so tracking might be misleading.
    If using one of the SB3 envs SB3DistributeRewardsWrapper can be used after logging.
    """

    def __init__(self, reward_func: RewardFunction, team_spirit=0.3):
        super().__init__()
        self.reward_func = reward_func
        self.team_spirit = team_spirit
        self.last_state = None
        self.base_rewards = {}
        self.avg_blue = 0
        self.avg_orange = 0

    def _compute(self, state: GameState, final=False):
        if state != self.last_state:
            self.base_rewards = {}
            sum_blue = 0
            n_blue = 0
            sum_orange = 0
            n_orange = 0
            for player in state.players:
                if final:
                    rew = self.reward_func.get_final_reward(player, state, None)
                else:
                    rew = self.reward_func.get_reward(player, state, None)

                self.base_rewards[player.car_id] = rew
                if player.team_num == BLUE_TEAM:
                    sum_blue += rew
                    n_blue += 1
                else:
                    sum_orange += rew
                    n_orange += 1
            self.avg_blue = sum_blue / (n_blue or 1)
            self.avg_orange = sum_orange / (n_orange or 1)

            self.last_state = state

    def _get_individual_reward(self, player):
        base_reward = self.base_rewards[player.car_id]
        if player.team_num == BLUE_TEAM:
            reward = self.team_spirit * self.avg_blue + (1 - self.team_spirit) * base_reward - self.avg_orange
        else:
            reward = self.team_spirit * self.avg_orange + (1 - self.team_spirit) * base_reward - self.avg_blue
        return reward

    def reset(self, initial_state: GameState):
        self.reward_func.reset(initial_state)

    def pre_step(self, state: GameState):
        self.reward_func.pre_step(state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self._compute(state, final=False)
        return self._get_individual_reward(player)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self._compute(state, final=True)
        return self._get_individual_reward(player)


class AnnealRewards(RewardFunction):

    def __init__(self, r_0, r_1, start, end):
        super(AnnealRewards, self).__init__()
        self.r_0 = r_0
        self.r_1 = r_1
        self.start = start
        self.end = end
        assert end >= start
        self.steps = end - start
        self.smd_config = SharedMemoryDict(name='shared_memory_dict', size=1024)

    def pre_step(self, state: GameState):
        self.r_0.pre_step(state)
        self.r_1.pre_step(state)

    def reset(self, initial_state: GameState):
        self.r_0.reset(initial_state)
        self.r_1.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        if self.smd_config["step"] < self.start:
            return self.r_0.get_reward(player, state, previous_action)
        elif self.smd_config["step"] > self.end:
            return self.r_1.get_reward(player, state, previous_action)
        else:

            r_0_reward = self.r_0.get_reward(player, state, previous_action)
            r_1_reward = self.r_1.get_reward(player, state, previous_action)

            frac = (self.smd_config["step"] - self.start) / self.steps

            r = frac * r_1_reward + (1 - frac) * r_0_reward

            return r
