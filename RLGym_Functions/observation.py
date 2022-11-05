import math
from typing import Any

import numpy as np
from SeerPPO import get_encoded_actionV2
from numba import jit
from rlgym.utils import common_values, ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState


def encode_all_players(player, state, inverted, demo_timers, ball):
    player_encoding = _encode_player(player, inverted, demo_timers.get(player), ball)

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
        encodings.append(_encode_player(p, inverted, demo_timers.get(p), ball))

    return encodings


def _encode_player(player: PlayerData, inverted: bool, demo_timer: float, ball):
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
        demo_timer * (1 / 3.0),
        player.boost_amount,
        player.on_ground,
        player.has_flip,
        demo_timer > 0,
        vel_norm * (1.0 / 6000.0),
        vel_norm > 2200,
        ball_diff_x * (1.0 / (4096.0 * 2.0)),
        ball_diff_y * (1.0 / (5120.0 * 2.0)),
        ball_diff_z * (1.0 / 2048.0),
        ball_diff_norm * (1.0 / 13272.55),
    ], dtype=np.float32)

    assert array.shape[0] == 23

    return array


def _encode_ball(ball):
    state = np.empty(10, dtype=np.float32)

    state[0] = ball.position[0] * (1.0 / 4096.0)
    state[1] = ball.position[1] * (1.0 / 5120.0)
    state[2] = ball.position[2] * (1.0 / 2048.0)
    state[3] = ball.linear_velocity[0] * (1.0 / 6000.0)
    state[4] = ball.linear_velocity[1] * (1.0 / 6000.0)
    state[5] = ball.linear_velocity[2] * (1.0 / 6000.0)
    state[6] = ball.angular_velocity[0] * (1.0 / 6.0)
    state[7] = ball.angular_velocity[1] * (1.0 / 6.0)
    state[8] = ball.angular_velocity[2] * (1.0 / 6.0)
    state[9] = np.linalg.norm([state[3], state[4], state[5]]) * (1.0 / 6000.0)

    assert state.shape[0] == 10

    return state


def encode_boost_pads(boost_pads_timers):
    pads_active = boost_pads_timers == 0.0

    pads_scaler = np.array([
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 10.0,
        1.0 / 10.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 10.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 10.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 10.0,
        1.0 / 10.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
    ], dtype=np.float32)

    boost_pads_timers *= pads_scaler

    return [pads_active, boost_pads_timers]


class SeerObsV2(ObsBuilder):
    def __init__(self, team_size, default_tick_skip=8.0, physics_ticks_per_second=120.0):
        super(SeerObsV2, self).__init__()

        self.team_size = team_size

        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.demo_timers = {}

        self.time_diff_tick = default_tick_skip / physics_ticks_per_second

    def reset(self, initial_state: GameState):

        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.demo_timers = {}

    def pre_step(self, state: GameState):

        self.update_boostpads(state.boost_pads)
        self.update_demo_timers(state)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = self.boost_pads_timers[::-1]

        else:
            inverted = False
            ball = state.ball
            pads = self.boost_pads_timers

        ball_data = _encode_ball(ball)
        player_encodings = encode_all_players(player, state, inverted, self.demo_timers, ball)

        pads_encoding = encode_boost_pads(pads)

        prev_action_enc = get_encoded_actionV2(previous_action)

        obs = np.concatenate([ball_data, prev_action_enc, *pads_encoding, *player_encodings])

        return obs

    def update_boostpads(self, pads):

        mask = pads == 1.0
        not_mask = np.logical_not(mask)

        self.boost_pads_timers[mask] = 0.0
        self.boost_pads_timers[not_mask] += self.time_diff_tick

    def update_demo_timers(self, state: GameState):

        for p in state.players:

            if p.is_demoed:
                self.demo_timers.update({p: self.demo_timers.get(p) + self.time_diff_tick})
            else:
                self.demo_timers.update({p: 0})
