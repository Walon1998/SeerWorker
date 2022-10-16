import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import BALL_RADIUS, BALL_MAX_SPEED, CAR_MAX_SPEED, CEILING_Z
from rlgym.utils.common_values import BLUE_TEAM
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards import SaveBoostReward, LiuDistancePlayerToBallReward, LiuDistanceBallToGoalReward, FaceBallReward, RewardIfClosestToBall, \
    AlignBallGoal, ConstantReward, RewardIfTouchedLast, RewardIfBehindBall, VelocityPlayerToBallReward, VelocityReward
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward


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


class GoalScoredReward(RewardFunction):
    def __init__(self, ball_speed_bonus=0.5):
        super(GoalScoredReward, self).__init__()
        self.prev_score_blue = 0
        self.prev_score_orange = 0
        self.prev_state_blue = GameState(None)
        self.prev_state_orange = GameState(None)
        self.ball_speed_bonus = ball_speed_bonus

    def reset(self, initial_state: GameState):
        self.prev_score_blue = initial_state.blue_score
        self.prev_score_orange = initial_state.orange_score
        self.prev_state_blue = initial_state
        self.prev_state_orange = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        if player.team_num == BLUE_TEAM:
            score = state.blue_score

            if score > self.prev_score_blue:
                bonus = self.ball_speed_bonus * np.linalg.norm(self.prev_state_blue.ball.linear_velocity) / BALL_MAX_SPEED
                self.prev_state_blue = state
                self.prev_score_blue = score
                return 1.0 + bonus
            else:
                self.prev_state_blue = state
                return 0.0

        else:
            score = state.orange_score

            if score > self.prev_score_orange:
                bonus = self.ball_speed_bonus * np.linalg.norm(self.prev_state_orange.ball.linear_velocity) / BALL_MAX_SPEED
                self.prev_score_orange = score
                self.prev_state_orange = state
                return 1.0 + bonus
            else:
                self.prev_state_orange = state
                return 0.0


class DemoReward(RewardFunction):
    def __init__(self):
        super(DemoReward, self).__init__()
        self.prev_demo_blue = 0
        self.prev_demo_orange = 0

    def reset(self, initial_state: GameState):
        for p in initial_state.players:

            if p.team_num == BLUE_TEAM:
                self.prev_demo_blue = p.match_demolishes
            else:
                self.prev_demo_orange = p.match_demolishes

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        demo = player.match_demolishes

        if player.team_num == BLUE_TEAM:

            if demo > self.prev_demo_blue:
                self.prev_demo_blue = demo
                return 1.0
            else:
                return 0.0

        else:

            if demo > self.prev_demo_orange:
                self.prev_demo_orange = demo
                return 1.0
            else:
                return 0.0


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


class SeerReward(RewardFunction):
    def __init__(self):
        super(RewardFunction, self).__init__()

        self.rewards = [GoalScoredReward(0.1),
                        DiffReward(SaveBoostReward(), 1.0),
                        SeerTouchBallReward(0.28361335653610786, 0.95, 0.1, 0.013),
                        DemoReward(),
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
            1.6,  # Goal Scored, Sparse, {0,1-1.5}
            0.1,  # Boost, Sparse, [0,1]
            0.05,  # Ball Touch, Sparse, [0,2]
            0.3,  # Demo, Sparse, {0,1}
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

        self.blue_score = 0.0
        self.orange_score = 0.0

    def reset(self, initial_state: GameState):

        for r in self.rewards:
            r.reset(initial_state)

        self.blue_score = 0.0
        self.orange_score = 0.0

    def pre_step(self, state: GameState):
        for p in state.players:
            rewards_list = [r.get_reward(p, state, None) for r in self.rewards]

            r = np.dot(rewards_list, self.weights)

            if p.team_num == BLUE_TEAM:

                self.blue_score = r
            else:
                self.orange_score = r

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            reward = self.blue_score - self.orange_score
        else:
            reward = self.orange_score - self.blue_score

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)
