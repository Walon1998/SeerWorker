import glob
from multiprocessing import Process, Queue
from typing import Union, Tuple

import gym as gym
import numpy as np
from gym import Wrapper, Env
import rlgym
from gym.spaces import Box
from rlgym.gamelaunch import LaunchPreference
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition, NoTouchTimeoutCondition

from RLGym_Functions.action import SeerAction
from RLGym_Functions.observation import SeerObs
from RLGym_Functions.reward import SeerReward
from RLGym_Functions.state_setter import SeerStateSetter


def worker(work_queue, result_queue, files, force_paging):
    env = rlgym.make(game_speed=100,
                     tick_skip=8,
                     spawn_opponents=True,
                     self_play=None,
                     team_size=1,
                     gravity=1,
                     boost_consumption=1,
                     terminal_conditions=[NoTouchTimeoutCondition(512), GoalScoredCondition()],
                     reward_fn=SeerReward(),
                     obs_builder=SeerObs(),
                     action_parser=SeerAction(),
                     state_setter=SeerStateSetter(files),
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

    def __init__(self, force_paging):
        super(AsyncEnv, self).__init__()

        self.result_queue = Queue()
        self.work_queue = Queue()

        # dummy_action_single = [2.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0] # Vollgas
        # self._dummy_action = np.array([dummy_action_single, dummy_action_single], dtype=np.float32)
        replays = glob.glob("./Replays/*.npz")

        p = Process(target=worker, args=(self.work_queue, self.result_queue, replays, force_paging))
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
