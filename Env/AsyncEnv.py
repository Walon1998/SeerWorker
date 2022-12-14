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
from rlgym.utils.state_setters import DefaultState, RandomState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition, NoTouchTimeoutCondition
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.hoops_setter import HoopsLikeSetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_terminals.game_condition import GameCondition

from SeerPPO.V2 import SeerActionV2
from SeerPPO.V2 import SeerObsV2
from RLGym_Functions.reward import SeerReward, DistributeRewardsV2, AnnealRewards
from RLGym_Functions.state_setter import SeerReplaySetterV2
from SeerPPO.V2 import SeerGameCondition


def worker(work_queue, result_queue, force_paging, team_size):
    condition = SeerGameCondition()

    env = rlgym.make(game_speed=100,
                     tick_skip=8,
                     spawn_opponents=True,
                     self_play=None,
                     team_size=team_size,
                     gravity=1,
                     boost_consumption=1,
                     terminal_conditions=[condition],
                     reward_fn=DistributeRewardsV2(SeerReward(condition)),
                     obs_builder=SeerObsV2(team_size, condition),
                     action_parser=SeerActionV2(),
                     state_setter=WeightedSampleSetter(
                         [SeerReplaySetterV2("./Replays/", team_size),
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
                     force_paging=force_paging,
                     raise_on_crash=False,
                     auto_minimize=True)

    result_queue.put(env.observation_space)
    result_queue.put(env.action_space)

    while True:
        id, action = work_queue.get()
        if id == 0:
            result_queue.put(env.reset())
        elif id == 1:
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            result_queue.put((obs, reward, done, info))


class AsyncEnv(gym.Env):

    def reset(self):
        return NotImplementedError()

    def step(self, action):
        return NotImplementedError()

    def __init__(self, team_size, force_paging):
        super(AsyncEnv, self).__init__()

        self.result_queue = Queue()
        self.work_queue = Queue()

        p = Process(target=worker, args=(self.work_queue, self.result_queue, force_paging, team_size))
        p.start()

        self.observation_space = self.result_queue.get()
        self.action_space = self.result_queue.get()

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
