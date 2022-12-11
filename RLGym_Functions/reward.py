import math
from typing import Union

import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import BALL_RADIUS, BALL_MAX_SPEED, CAR_MAX_SPEED, CEILING_Z
from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards import SaveBoostReward, LiuDistancePlayerToBallReward, LiuDistanceBallToGoalReward, FaceBallReward, RewardIfClosestToBall, \
    AlignBallGoal, ConstantReward, RewardIfTouchedLast, RewardIfBehindBall, VelocityPlayerToBallReward, VelocityReward, EventReward
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
from shared_memory_dict import SharedMemoryDict


class SeerKickoffReward(RewardFunction):
    """
    a simple reward that encourages driving towards the ball fast while it's in the neutral kickoff position
    """

    def __init__(self):
        super().__init__()
        self.vel_dir_reward = VelocityPlayerToBallReward()

    def reset(self, initial_state: GameState):
        self.vel_dir_reward.reset(initial_state)

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action
    ) -> float:
        reward = 0

        dist = np.linalg.norm(player.car_data.position - state.ball.position)

        for p in state.players:
            if p.car_id == player.car_id or player.team_num != p.team_num:
                continue

            dist2 = np.linalg.norm(p.car_data.position - state.ball.position)
            if dist2 < dist:
                return 0

        if state.ball.position[0] == 0 and state.ball.position[1] == 0:
            reward += self.vel_dir_reward.get_reward(player, state, previous_action)
        return reward


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


class ForwardVelocity(RewardFunction):
    def __init__(self):
        super(ForwardVelocity, self).__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel_norm = player.car_data.linear_velocity / CAR_MAX_SPEED
        return float(np.dot(player.car_data.forward(), vel_norm))


class SeerRewardV2(RewardFunction):
    def __init__(self):
        super(RewardFunction, self).__init__()

        self.rewards = [EventReward(goal=0.0, team_goal=1.0, touch=0.05, demo=0.3, boost_pickup=0.1),
                        LiuDistancePlayerToBallReward(),
                        LiuDistanceBallToGoalReward(False),
                        FaceBallReward(),
                        AlignBallGoal(0.5, 0.5),
                        RewardIfClosestToBall(ConstantReward(), False),
                        RewardIfTouchedLast(ConstantReward()),
                        RewardIfBehindBall(ConstantReward()),
                        VelocityPlayerToBallReward(False),
                        KickoffReward(),
                        VelocityReward(False),
                        SaveBoostReward(),
                        ForwardVelocity(),
                        AirReward()]

        self.weights = np.array([
            1.0,  # Event reward
            0.0025,  # Distance Ball Player, Cont., [0,1]
            0.0025,  # Distance Ball Goal, Cont., [0,1]
            0.000625,  # Face Ball, Cont., [-1,1]
            0.0025,  # Align ball goal, cont, [-1,1]
            0.00125,  # Closest to ball, cont, {0,1}
            0.00125,  # Touched Last, cont, {0,1}
            0.00125,  # Behind Ball, cont, {0,1},
            0.0025,  # velocity to ball, cont, [0,1]
            0.01,  # Kickoff, cont, [0,1]
            0.000625,  # velocity, cont, [0,1]
            0.00125,  # Save Boost, cont, [0,1]
            0.0015,  # forward velocity, cont, [-1,1]
            0.000125,  # Air Reward, Cont., [0,1]
        ], dtype=np.float32)

        assert len(self.weights) == len(self.rewards)

    def reset(self, initial_state: GameState):

        for r in self.rewards:
            r.reset(initial_state)

    def pre_step(self, state: GameState):

        for r in self.rewards:
            r.pre_step(state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        rewards_list = [r.get_reward(player, state, previous_action) for r in self.rewards]
        return np.dot(rewards_list, self.weights)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)


class SeerRewardV2_1(SeerRewardV2):
    def __init__(self):
        super(SeerRewardV2, self).__init__()

        self.rewards = [EventReward(goal=0.0, team_goal=1.25, touch=0.025, demo=0.3, boost_pickup=0.1),
                        LiuDistancePlayerToBallReward(),
                        LiuDistanceBallToGoalReward(False),
                        FaceBallReward(),
                        AlignBallGoal(0.5, 0.5),
                        RewardIfClosestToBall(ConstantReward(), False),
                        RewardIfTouchedLast(ConstantReward()),
                        RewardIfBehindBall(ConstantReward()),
                        VelocityPlayerToBallReward(False),
                        SeerKickoffReward(),
                        VelocityReward(False),
                        SaveBoostReward(),
                        ForwardVelocity(),
                        AirReward()]

        self.weights = np.array([
            1.0,  # Event reward
            0.0025,  # Distance Ball Player, Cont., [0,1]
            0.0025,  # Distance Ball Goal, Cont., [0,1]
            0.000625,  # Face Ball, Cont., [-1,1]
            0.0025,  # Align ball goal, cont, [-1,1]
            0.00125,  # Closest to ball, cont, {0,1}
            0.00125,  # Touched Last, cont, {0,1}
            0.00125,  # Behind Ball, cont, {0,1},
            0.0025,  # velocity to ball, cont, [0,1]
            0.01,  # Kickoff, cont, [0,1]
            0.000625,  # velocity, cont, [0,1]
            0.00125,  # Save Boost, cont, [0,1]
            0.0015,  # forward velocity, cont, [-1,1]
            0.000125,  # Air Reward, Cont., [0,1]
        ], dtype=np.float32)

        assert len(self.weights) == len(self.rewards)


class SeerRewardV2_2(SeerRewardV2):
    def __init__(self):
        super(SeerRewardV2, self).__init__()

        self.rewards = [EventReward(goal=0.0, team_goal=1.5, touch=0.025, demo=0.3, boost_pickup=0.1),
                        LiuDistancePlayerToBallReward(),
                        LiuDistanceBallToGoalReward(False),
                        FaceBallReward(),
                        AlignBallGoal(0.5, 0.5),
                        RewardIfClosestToBall(ConstantReward(), False),
                        RewardIfTouchedLast(ConstantReward()),
                        RewardIfBehindBall(ConstantReward()),
                        VelocityPlayerToBallReward(False),
                        SeerKickoffReward(),
                        VelocityReward(False),
                        SaveBoostReward(),
                        ForwardVelocity(),
                        AirReward()]

        self.weights = np.array([
            1.0,  # Event reward
            0.0025,  # Distance Ball Player, Cont., [0,1]
            0.0025,  # Distance Ball Goal, Cont., [0,1]
            0.000625,  # Face Ball, Cont., [-1,1]
            0.0025,  # Align ball goal, cont, [-1,1]
            0.00125,  # Closest to ball, cont, {0,1}
            0.00125,  # Touched Last, cont, {0,1}
            0.00125,  # Behind Ball, cont, {0,1},
            0.0025,  # velocity to ball, cont, [0,1]
            0.01,  # Kickoff, cont, [0,1]
            0.000625,  # velocity, cont, [0,1]
            0.00125,  # Save Boost, cont, [0,1]
            0.0015,  # forward velocity, cont, [-1,1]
            0.000125,  # Air Reward, Cont., [0,1]
        ], dtype=np.float32)

        assert len(self.weights) == len(self.rewards)


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
