import glob
import os.path
import random

import numpy as np
from rlgym.utils import common_values
from rlgym.utils.state_setters import StateSetter, DefaultState
from rlgym.utils.state_setters import StateWrapper


class SeerReplaySetterv2(StateSetter):
    def __init__(self, dir, team_size):
        super().__init__()
        self.dir = dir
        self.team_size = team_size

        if self.team_size == 1:
            dir = os.path.join(dir, "1v1")
        elif self.team_size == 2:
            dir = os.path.join(dir, "2v2")
        elif self.team_size == 3:
            dir = os.path.join(dir, "3v3")
        else:
            raise NotImplementedError

        self.files = glob.glob(os.path.join(dir, "*.npz"))

    def _set_ball(self, ball, ball_array):
        ball.set_pos(*ball_array[0:3])
        ball.set_lin_vel(*ball_array[3:6])
        ball.set_ang_vel(*ball[6:9])

    def _set_car(self, car, car_array):
        car.set_pos(*car_array[0:3])
        car.set_lin_vel(*car_array[3:6])
        car.set_rot(*car_array[6:9])
        car.set_ang_vel(*car_array[9:12])
        car.boost = car_array[13] / 100.0

    def reset(self, state_wrapper: StateWrapper):

        try:
            f = random.choice(self.files)
            array = np.load(f)["data"]
            index = random.choice(range(0, int(array.shape[0])))
            slice = array[index]
        except Exception as e:
            print(e, f)
            DefaultState().reset(state_wrapper)
            return

        self._set_ball(state_wrapper.ball, slice[0:12])

        blue_cars = []
        orange_cars = []

        for car in state_wrapper.cars:
            if car.team_num == common_values.BLUE_TEAM:
                blue_cars.append(car)
            else:
                orange_cars.append(car)

        start = 12
        size = 23

        for car in blue_cars + orange_cars:
            self._set_car(car, slice[start:start + size])
            start += size
