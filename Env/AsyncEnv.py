import glob
from multiprocessing import Process, Queue
from typing import Union, Tuple

import gym as gym
import numpy as np
from gym import Wrapper, Env
import rlgym
from gym.spaces import Box
from rlgym.gamelaunch import LaunchPreference
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward
from rlgym.utils.state_setters import DefaultState, RandomState
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.hoops_setter import HoopsLikeSetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter

from Env.DelayWrapper import DelayWrapper
from RLGym_Functions.action import SeerAction
from RLGym_Functions.observation import SeerObs
from RLGym_Functions.reward import SeerReward
from RLGym_Functions.state_setter import SeerReplaySetter


def worker(work_queue, result_queue, files):
    env = rlgym.make(game_speed=100,
                     tick_skip=8,
                     spawn_opponents=True,
                     self_play=None,
                     team_size=1,
                     gravity=1,
                     boost_consumption=1,
                     terminal_conditions=[TimeoutCondition(512), GoalScoredCondition()],
                     reward_fn=DefaultReward(),
                     obs_builder=SeerObs(),
                     action_parser=SeerAction(),
                     state_setter=WeightedSampleSetter(
                         [SeerReplaySetter(files),
                          DefaultState(),
                          GoaliePracticeState(),
                          HoopsLikeSetter(),
                          WallPracticeState(),
                          RandomState(ball_rand_speed=False, cars_rand_speed=False, cars_on_ground=True),
                          RandomState(ball_rand_speed=True, cars_rand_speed=False, cars_on_ground=True),
                          RandomState(ball_rand_speed=True, cars_rand_speed=True, cars_on_ground=False),
                          ]
                         , [0.7,
                            0.1,
                            0.05,
                            0.05,
                            0.05,
                            0.01,
                            0.02,
                            0.02,
                            ]),
                     launch_preference=LaunchPreference.EPIC,
                     use_injector=True,
                     force_paging=True,
                     raise_on_crash=False,
                     auto_minimize=True)

    env = DelayWrapper(env)

    result_queue.put(env.observation_space)
    result_queue.put(env.action_space)

    while True:
        id, action = work_queue.get()
        if id == 0:
            result_queue.put(env.reset())
        elif id == 1:
            result_queue.put(env.step(action))


class AsyncEnv(gym.Env):

    def reset(self):
        return NotImplementedError()

    def step(self, action):
        return NotImplementedError()

    def __init__(self):
        super(AsyncEnv, self).__init__()

        self.result_queue = Queue()
        self.work_queue = Queue()

        # dummy_action_single = [2.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0] # Vollgas
        # self._dummy_action = np.array([dummy_action_single, dummy_action_single], dtype=np.float32)
        replays = glob.glob("./Replays/*.npz")

        p = Process(target=worker, args=(self.work_queue, self.result_queue, replays))
        p.start()

        self.observation_space = self.result_queue.get()
        self.action_space = self.result_queue.get()

        # self.result_queue.put((None, None, None, None))

    def reset_put(self):
        self.work_queue.put((0, None))

    def reset_get(self):
        return self.result_queue.get()

    def step_put(self, action):
        self.work_queue.put((1, action))

    def step_get(self):
        return self.result_queue.get()

    def render(self, mode="human"):
        pass
