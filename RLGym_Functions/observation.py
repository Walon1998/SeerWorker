from typing import Any, Tuple

import numpy as np
from rlgym.utils import common_values, ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState


def get_distance(array_0: np.ndarray, array_1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # assert array_0.shape[0] == 3
    # assert array_1.shape[0] == 3

    diff = array_0 - array_1

    norm = np.linalg.norm(diff)

    return diff, np.array(norm, dtype=np.float32).reshape(1)


def get_speed(array: np.ndarray):
    speed = np.linalg.norm(array)

    is_super_sonic = speed >= 2200.0

    return np.array(speed, dtype=np.float32).reshape(1), np.array(is_super_sonic, dtype=np.float32).reshape(1)


def impute_features(player_car_state: np.ndarray, opponent_car_data: np.ndarray, pads, ball_data: np.ndarray, prev_action_enc: np.ndarray):
    # assert x_train.shape[0] == input_features_replay

    player_0 = player_car_state
    player_1 = opponent_car_data
    boost_pads_timer = pads
    ball = ball_data

    # assert player_0.shape[0] == 16
    # assert player_1.shape[0] == 16
    # assert ball.shape[0] == 9

    player_0_pos = player_0[0:3]
    # player_0_rotation = player_0[:, 3:6]
    player_0_velocity = player_0[6:9]
    # player_0_ang_velocity = player_0[:, 9:12]
    player_0_demo_timer = player_0[12]

    player_1_pos = player_1[0:3]
    # player_1_rotation = player_1[:, 3:6]
    player_1_velocity = player_1[6:9]
    # player_1_ang_velocity = player_1[:, 9:12]
    player_1_demo_timer = player_1[12]

    ball_pos = ball[0:3]
    ball_velocity = ball[3:6]
    # ball_1_ang_velocity = ball[:, 6:9]

    is_boost_active = boost_pads_timer == 0.0
    player_0_is_alive = player_0_demo_timer == 0.0
    player_1_is_alive = player_1_demo_timer == 0.0

    player_0_is_alive = np.array(player_0_is_alive, dtype=np.float32).reshape(1)
    player_1_is_alive = np.array(player_1_is_alive, dtype=np.float32).reshape(1)

    player_0_speed, player_0_super_sonic = get_speed(player_0_velocity)
    player_1_speed, player_1_super_sonic = get_speed(player_1_velocity)
    ball_speed, _ = get_speed(ball_velocity)

    player_opponent_pos_diff, player_opponent_pos_norm = get_distance(player_0_pos, player_1_pos)
    player_opponent_vel_diff, player_opponent_vel_norm = get_distance(player_0_velocity, player_1_velocity)

    player_ball_pos_diff, player_ball_pos_norm = get_distance(player_0_pos, ball_pos)
    player_ball_vel_diff, player_ball_vel_norm = get_distance(player_0_velocity, ball_velocity)

    opponent_ball_pos_diff, opponent_ball_pos_norm = get_distance(player_1_pos, ball_pos)
    opponent_ball_vel_diff, opponent_ball_vel_norm = get_distance(player_1_velocity, ball_velocity)

    result = np.concatenate((
        player_car_state, opponent_car_data, pads, ball_data,
        player_opponent_pos_diff, player_opponent_pos_norm,
        player_opponent_vel_diff, player_opponent_vel_norm,
        player_ball_pos_diff, player_ball_pos_norm,
        player_ball_vel_diff, player_ball_vel_norm,
        opponent_ball_pos_diff, opponent_ball_pos_norm,
        opponent_ball_vel_diff, opponent_ball_vel_norm,
        is_boost_active,
        player_0_is_alive, player_1_is_alive,
        player_0_speed, player_0_super_sonic,
        player_1_speed, player_1_super_sonic,
        ball_speed, prev_action_enc)
    )

    return result


def _encode_player(player: PlayerData, inverted: bool, demo_timer: float):
    if inverted:
        player_car = player.inverted_car_data
    else:
        player_car = player.car_data

    array = np.array([
        player_car.position[0],
        player_car.position[1],
        player_car.position[2],
        player_car.pitch(),
        player_car.yaw(),
        player_car.roll(),
        player_car.linear_velocity[0],
        player_car.linear_velocity[1],
        player_car.linear_velocity[2],
        player_car.angular_velocity[0],
        player_car.angular_velocity[1],
        player_car.angular_velocity[2],
        demo_timer,
        player.boost_amount * 100,
        player.on_ground,
        player.has_flip,
    ], dtype=np.float32)

    assert array.shape[0] == 16

    return array


def _encode_ball(ball):
    state = np.empty(9, dtype=np.float32)

    state[0] = ball.position[0]
    state[1] = ball.position[1]
    state[2] = ball.position[2]
    state[3] = ball.linear_velocity[0]
    state[4] = ball.linear_velocity[1]
    state[5] = ball.linear_velocity[2]
    state[6] = ball.angular_velocity[0]
    state[7] = ball.angular_velocity[1]
    state[8] = ball.angular_velocity[2]

    assert state.shape[0] == 9

    return state


def get_encoded_action(action: np.ndarray) -> np.ndarray:
    # throttle, steer, pitch, yaw, roll, jump, boost, handbrake

    action_encoding = np.zeros(19, dtype=np.float32)

    acc = 0
    throttle_index = action[0] + 1 + acc
    acc += 3
    steer_yaw_index = (action[1] + 1) * 2 + acc
    acc += 5
    pitch_index = (action[2] + 1) * 2 + acc
    acc += 5
    roll_index = action[4] + 1 + acc

    action_encoding[int(throttle_index)] = 1.0
    action_encoding[int(steer_yaw_index)] = 1.0
    action_encoding[int(pitch_index)] = 1.0
    action_encoding[int(roll_index)] = 1.0

    action_encoding[18] = action[7]
    action_encoding[17] = action[6]
    action_encoding[16] = action[5]

    # action_without_yaw = action[[0, 1, 2, 4, 5, 6, 7]]  # remove yaw

    # encoder = OneHotEncoder(sparse=False, drop='if_binary',
    #                         categories=[np.array([-1., 0., 1.]), np.array([-1., -0.5, 0., 0.5, 1.]), np.array([-1., -0.5, 0., 0.5, 1.]), np.array([-1., 0., 1.]), np.array([0., 1.]),
    #                                     np.array([0., 1.]),
    #                                     np.array([0., 1.])])

    return action_encoding


class SeerObs(ObsBuilder):
    def __init__(self, default_tick_skip=8.0, physics_ticks_per_second=120.0):
        super(SeerObs, self).__init__()

        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.blue_demo_timer = 0.0
        self.orange_demo_timer = 0.0

        self.time_diff_tick = default_tick_skip / physics_ticks_per_second

    def reset(self, initial_state: GameState):

        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.blue_demo_timer = 0.0
        self.orange_demo_timer = 0.0

    def pre_step(self, state: GameState):

        self.update_boostpads(state.boost_pads)
        self.update_demo_timers(state)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        assert len(state.players) == 2

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = self.boost_pads_timers[::-1]
            player_demo_timer = self.orange_demo_timer
            opponent_demo_timer = self.blue_demo_timer

        else:
            inverted = False
            ball = state.ball
            pads = self.boost_pads_timers
            opponent_demo_timer = self.orange_demo_timer
            player_demo_timer = self.blue_demo_timer

        ball_data = _encode_ball(ball)
        player_car_state = _encode_player(player, inverted, player_demo_timer)

        # opponent_car_data = None
        for other in state.players:
            if other.car_id == player.car_id:
                continue

            opponent_car_data = _encode_player(other, inverted, opponent_demo_timer)

        # assert opponent_car_data is not None

        prev_action_enc = get_encoded_action(previous_action)

        x_train = impute_features(player_car_state, opponent_car_data, pads, ball_data, prev_action_enc)

        return x_train

    def update_boostpads(self, pads):

        mask = pads == 1.0
        not_mask = np.logical_not(mask)

        self.boost_pads_timers[mask] = 0.0
        self.boost_pads_timers[not_mask] += self.time_diff_tick

    def update_demo_timers(self, state: GameState):

        for p in state.players:

            if p.team_num == common_values.ORANGE_TEAM:

                if p.is_demoed:
                    self.orange_demo_timer += self.time_diff_tick
                else:
                    self.orange_demo_timer = 0.0
            else:
                if p.is_demoed:
                    self.blue_demo_timer += self.time_diff_tick
                else:
                    self.blue_demo_timer = 0.0
