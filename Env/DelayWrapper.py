import threading
from queue import Queue

import numpy as np
from gym import Wrapper


def worker(env, work_queue, result_queue):
    while True:
        id, action = work_queue.get()
        if id == 0:
            result_queue.put(env.reset())
        elif id == 1:
            result_queue.put(env.step(action))


class DelayWrapper(Wrapper):
    def __init__(self, env):
        super(DelayWrapper, self).__init__(env)

        self.work_queue = Queue()
        self.result_queue = Queue()

        t = threading.Thread(target=worker, args=(env, self.work_queue, self.result_queue,))
        t.start()

        dummy_action_single = [2.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0]  # vollgas
        self._dummy_action = np.array([dummy_action_single, dummy_action_single], dtype=np.float32)

        self.result_queue.put((None, None, None, None))

    def reset(self, ):
        _, _, _, _ = self.result_queue.get()
        self.work_queue.put((0, None))
        obs = self.result_queue.get()
        self.work_queue.put((1, self._dummy_action))
        return obs

    def step(self, action):
        obs, reward, done, info = self.result_queue.get()
        self.work_queue.put((1, action))
        if done:
            obs = self.reset()
        return obs, reward, done, info
