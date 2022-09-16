from typing import Any, Tuple

import numpy as np
from rlgym.utils import common_values, ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
from SeerPPO import impute_features, get_encoded_action


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
