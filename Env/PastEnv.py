import asyncio
import glob
import os
import time
from multiprocessing import Queue, Process

import aiohttp
import gym
import numpy as np
import requests
import torch
from tqdm import tqdm

from Env.MulitEnv import MultiEnv
from SeerPPO import SeerNetwork, RolloutBuffer

from Env.NormalizeReward import NormalizeReward


async def download_models(session, url, disable=False):
    print("Downloading new models!")
    local_files = [os.path.basename(x) for x in glob.glob("./Models/*.pt")]
    async with session.get(url + "/pastmodels") as resp:
        assert resp.status == 200
        files = await resp.read()
        files = files.decode('ascii').split(",")
        file_to_get = set(files) - set(local_files)
        for f in tqdm(file_to_get, disable=disable):
            async with session.get(url + "/Models/{f}".format(f=f)) as resp2:
                assert resp2.status == 200
                with open("./Models/{f}".format(f=f), "wb") as newfile:
                    newfile.write(await resp2.read())


async def past_model_downloader_async(url):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0, connect=60, sock_connect=60, sock_read=60)) as session:
        while True:
            time.sleep(5 * 60)
            await download_models(session, url, disable=True)


def past_model_downloader(url):
    asyncio.run(past_model_downloader_async(url))


def get_past_models(session, url):
    return ["2.pt"]  # TODO


def choose_model(past_models):
    return "./Models/" + past_models[0]  # TODO


def past_worker(work_queue, result_queue, batch_size, device, url):
    session = requests.Session()
    past_models = get_past_models(session, url)
    lstm_state = torch.zeros(1, batch_size, 512, device=device, requires_grad=False), torch.zeros(1, batch_size, 512, device=device, requires_grad=False)
    policy = SeerNetwork()
    policy.load_state_dict(torch.load(choose_model(past_models)))
    policy.to(device)

    counter = 0

    while True:
        obs, episode_starts = work_queue.get()
        obs = torch.tensor(obs, dtype=torch.float32).to(device, non_blocking=True)
        episode_starts = torch.tensor(episode_starts, dtype=torch.float32).to(device, non_blocking=True)

        if device == "cuda":
            torch.cuda.synchronize(device="cuda")

        actions, lstm_state = policy.predict_actions(obs, lstm_state, episode_starts, True)

        result_queue.put(actions.to("cpu").numpy())

        counter += 1

        if counter % 5_000 == 0:
            past_models = get_past_models(session, url)
            policy.load_state_dict(torch.load(choose_model(past_models), map_location=device))
            lstm_state = torch.zeros(1, batch_size, 512, device=device, requires_grad=False), torch.zeros(1, batch_size, 512, device=device, requires_grad=False)


class PastEnv(gym.Env):

    def __init__(self, num_instances, old_instances, mean, var, gamma, device, url):
        super(PastEnv, self).__init__()

        self.num_instances = num_instances
        self.old_instances = old_instances

        assert self.num_instances >= self.old_instances

        self.obs_shape = (self.num_instances * 2, 159)
        self.reward_shape = (self.num_instances * 2)

        self.num_envs = 2 * self.num_instances - old_instances

        self.mask_old_opponents = np.zeros(2 * self.num_instances, dtype=np.bool)
        for i in range(self.old_instances):
            self.mask_old_opponents[2 * i] = True

        self.mask_new_opponent = np.logical_not(self.mask_old_opponents)

        self.env = MultiEnv(num_instances)
        self.env = NormalizeReward(self.env, mean, var, gamma=gamma)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.work_queue = Queue()
        self.result_queue = Queue()
        p = Process(target=past_worker, args=(self.work_queue, self.result_queue, self.old_instances, device, url))
        p.start()

    def reset(self):
        obs = self.env.reset()

        obs_old = obs[self.mask_old_opponents]
        obs_new = obs[self.mask_new_opponent]

        self.work_queue.put((obs_old, np.ones(self.old_instances)))
        return obs_new

    def step(self, action):
        old_action = self.result_queue.get()

        combined_actions = np.empty([self.num_instances * 2, 7], dtype=np.float32)
        combined_actions[self.mask_old_opponents] = old_action
        combined_actions[self.mask_new_opponent] = action
        obs, rew, done, info = self.env.step(combined_actions)

        obs_old = obs[self.mask_old_opponents]
        obs_new = obs[self.mask_new_opponent]

        # rew_old = rew[self.mask_old_opponents]
        rew_new = rew[self.mask_new_opponent]

        done_old = done[self.mask_old_opponents]
        done_new = done[self.mask_new_opponent]

        self.work_queue.put((obs_old, done_old))

        return obs_new, rew_new, done_new, None

    def render(self, mode="human"):
        pass
