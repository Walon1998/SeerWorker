import os.path
import random

import numpy as np
from rlgym.utils import common_values
from rlgym.utils.state_setters import StateSetter, DefaultState, RandomState
from rlgym.utils.state_setters import StateWrapper
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.hoops_setter import HoopsLikeSetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter


class SeerStateSetter(StateSetter):

    def __init__(self, files):
        super(SeerStateSetter, self).__init__()
        self.files = files

        self.r = WeightedSampleSetter(
            [SeerReplaySetter(files),
             DefaultState(),
             GoaliePracticeState(),
             HoopsLikeSetter(),
             WallPracticeState(),
             RandomState(ball_rand_speed=False, cars_rand_speed=False, cars_on_ground=True),
             RandomState(ball_rand_speed=True, cars_rand_speed=False, cars_on_ground=True),
             RandomState(ball_rand_speed=True, cars_rand_speed=True, cars_on_ground=False),
             ]
            , [0.65,
               0.2,
               0.05,
               0.05,
               0.05,
               0.00,
               0.00,
               0.00,
               ])

    def reset(self, state_wrapper: StateWrapper):
        self.r.reset(state_wrapper)


class SeerReplaySetter(StateSetter):
    def __init__(self, files):
        super().__init__()
        self.files = files

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected frame from replay.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        # possible kickoff indices are shuffled

        f = random.choice(self.files)

        try:
            array = np.load(f)
            x_train = random.choice([array["x_train_winner"], array["x_train_looser"]])
            index = random.choice(range(0, int(x_train.shape[0] * 0.75)))
            slice = x_train[index]
        except Exception as e:
            print(e, f)
            DefaultState().reset(state_wrapper)
            return

        player_0 = slice[0:16]
        player_1 = slice[16:32]
        # boost_pads_timer = x_train[:, 32:66]
        ball = slice[66:75]

        assert player_0.shape[0] == 16
        assert player_1.shape[0] == 16
        assert ball.shape[0] == 9

        player_0_pos = player_0[0:3]
        player_0_rotation = player_0[3:6]
        player_0_velocity = player_0[6:9]
        player_0_ang_velocity = player_0[9:12]
        # player_0_demo_timer = player_0[:, 12]
        player_0_boost = player_0[13]

        player_1_pos = player_1[0:3]
        player_1_rotation = player_1[3:6]
        player_1_velocity = player_1[6:9]
        player_1_ang_velocity = player_1[9:12]
        # player_1_demo_timer = player_1[:, 12]
        player_1_boost = player_1[13]

        ball_pos = ball[0:3]
        ball_velocity = ball[3:6]
        ball_ang_velocity = ball[6:9]

        state_wrapper.ball.set_pos(*ball_pos)
        state_wrapper.ball.set_lin_vel(*ball_velocity)
        state_wrapper.ball.set_ang_vel(*ball_ang_velocity)

        for car in state_wrapper.cars:

            if car.team_num == common_values.BLUE_TEAM:

                car.set_pos(*player_0_pos)
                car.set_lin_vel(*player_0_velocity)
                car.set_rot(*player_0_rotation)
                car.set_ang_vel(*player_0_ang_velocity)
                car.boost = player_0_boost / 100.0
            else:

                car.set_pos(*player_1_pos)
                car.set_lin_vel(*player_1_velocity)
                car.set_rot(*player_1_rotation)
                car.set_ang_vel(*player_1_ang_velocity)
                car.boost = player_1_boost / 100.0
