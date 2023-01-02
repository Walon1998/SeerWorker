import math
from typing import Any
import numpy as np
from numba import jit
from rlgym.utils import common_values, ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState


@jit(nopython=True, fastmath=True)
def get_encoded_actionV2(action: np.ndarray) -> np.ndarray:
    # throttle, steer, pitch, yaw, roll, jump, boost, handbrake

    action_encoding = np.zeros(15, dtype=np.float32)

    acc = 0
    throttle_index = action[0] + 1 + acc
    acc += 3
    steer_yaw_index = action[1] + 1 + acc
    acc += 3
    pitch_index = action[2] + 1 + acc
    acc += 3
    roll_index = action[4] + 1 + acc

    action_encoding[int(throttle_index)] = 1.0
    action_encoding[int(steer_yaw_index)] = 1.0
    action_encoding[int(pitch_index)] = 1.0
    action_encoding[int(roll_index)] = 1.0

    action_encoding[12] = action[5]
    action_encoding[13] = action[6]
    action_encoding[14] = action[7]

    return action_encoding


def encode_all_players(player, state, inverted, ball):
    player_encoding = _encode_player(player, inverted, ball)

    same_team = []
    opponent_team = []

    for p in state.players:

        if p == player:
            continue

        if p.team_num == player.team_num:
            same_team.append(p)
        else:
            opponent_team.append(p)

    # assert len(opponent_team) > len(same_team)

    same_team.sort(key=lambda x: x.car_id)
    opponent_team.sort(key=lambda x: x.car_id)

    encodings = [player_encoding]
    for p in same_team + opponent_team:
        encodings.append(_encode_player(p, inverted, ball))

    return encodings


def _encode_player(player: PlayerData, inverted: bool, ball):
    if inverted:
        player_car = player.inverted_car_data
    else:
        player_car = player.car_data

    vel_norm = np.linalg.norm([player_car.linear_velocity[0],
                               player_car.linear_velocity[1],
                               player_car.linear_velocity[2]])

    ball_diff_x = ball.position[0] - player_car.position[0]
    ball_diff_y = ball.position[1] - player_car.position[1]
    ball_diff_z = ball.position[2] - player_car.position[2]
    ball_diff_norm = np.linalg.norm([ball_diff_x, ball_diff_y, ball_diff_z])

    array = np.array([
        player_car.position[0] * (1.0 / 4096.0),
        player_car.position[1] * (1.0 / 5120.0),
        player_car.position[2] * (1.0 / 2048.0),
        player_car.pitch() * (1.0 / math.pi),
        player_car.yaw() * (1.0 / math.pi),
        player_car.roll() * (1.0 / math.pi),
        player_car.linear_velocity[0] * (1.0 / 2300.0),
        player_car.linear_velocity[1] * (1.0 / 2300.0),
        player_car.linear_velocity[2] * (1.0 / 2300.0),
        player_car.angular_velocity[0] * (1.0 / 5.5),
        player_car.angular_velocity[1] * (1.0 / 5.5),
        player_car.angular_velocity[2] * (1.0 / 5.5),
        player.is_demoed,
        player.boost_amount,
        player.on_ground,
        player.has_flip,
        vel_norm * (1.0 / 6000.0),
        vel_norm > 2200,
        ball_diff_x * (1.0 / (4096.0 * 2.0)),
        ball_diff_y * (1.0 / (5120.0 * 2.0)),
        ball_diff_z * (1.0 / 2048.0),
        ball_diff_norm * (1.0 / 13272.55),
    ], dtype=np.float32)

    assert array.shape[0] == 22

    return array


def _encode_ball(ball):
    state = np.array([
        ball.position[0] * (1.0 / 4096.0),
        ball.position[1] * (1.0 / 5120.0),
        ball.position[2] * (1.0 / 2048.0),
        ball.linear_velocity[0] * (1.0 / 6000.0),
        ball.linear_velocity[1] * (1.0 / 6000.0),
        ball.linear_velocity[2] * (1.0 / 6000.0),
        ball.angular_velocity[0] * (1.0 / 6.0),
        ball.angular_velocity[1] * (1.0 / 6.0),
        ball.angular_velocity[2] * (1.0 / 6.0),
        np.linalg.norm([ball.linear_velocity[0], ball.linear_velocity[1], ball.linear_velocity[2]]) * (1.0 / 6000.0),
    ], dtype=np.float32)

    assert state.shape[0] == 10

    return state


def encode_game_state(condition, inverted):
    dif, timer, overtime = condition.score, condition.timer, condition.overtime

    if inverted:
        dif *= -1

    game_state = [dif / 10, timer / 300, overtime]

    return game_state


class SeerObsV2(ObsBuilder):
    def __init__(self, team_size, condition):
        super(SeerObsV2, self).__init__()
        self.team_size = team_size
        self.condition = condition

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads

        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        ball_data = _encode_ball(ball)
        player_encodings = encode_all_players(player, state, inverted, ball)
        prev_action_enc = get_encoded_actionV2(previous_action)
        game_state = encode_game_state(self.condition, inverted)
        obs = np.concatenate([*player_encodings, ball_data, prev_action_enc, pads, game_state])

        return obs
