import os
import pickle
import random
from multiprocessing import Queue, Process

import gym
import numpy as np
import requests
import torch
from SeerPPO.past_models_download import start_past_model_downloader

from Env.MonitorWrapper import MonitorWrapper
from Env.MulitEnv import MultiEnv, get_obs_size
from SeerPPO.V2 import SeerNetworkV2

from Env.NormalizeReward import NormalizeReward


def get_past_models(session, url):
    respone = session.get(url + "/pastopponent")
    assert respone.status_code == 200
    data = pickle.loads(respone.content)
    return data


def choose_model(past_models):
    while True:
        f = random.choice(past_models)
        filename = "./Models/" + f
        if os.path.isfile(filename):
            return filename


def past_worker(work_queue, result_queue, batch_size, device, url):
    session = requests.Session()
    past_models = get_past_models(session, url)
    policy = SeerNetworkV2()
    lstm_state = torch.zeros(1, batch_size, policy.LSTM.hidden_size, device=device, requires_grad=False, dtype=torch.float32), torch.zeros(1, batch_size, policy.LSTM.hidden_size, device=device,
                                                                                                                                           requires_grad=False, dtype=torch.float32)

    policy.load_state_dict(torch.load(choose_model(past_models)))

    policy.to(device)

    counter = 0

    with torch.no_grad():

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
                lstm_state = torch.zeros(1, batch_size, policy.LSTM.hidden_size, device=device, requires_grad=False, dtype=torch.float32), torch.zeros(1, batch_size, policy.LSTM.hidden_size,
                                                                                                                                                       device=device, requires_grad=False,
                                                                                                                                                       dtype=torch.float32)


class PastEnv(gym.Env):

    def __init__(self, env, old_instances, device, url):
        super(PastEnv, self).__init__()

        self.env = env
        self.old_instances = old_instances
        self.num_instances = env.num_instances
        self.team_size = env.team_size
        self.is_vector_env = True

        assert self.num_instances >= self.old_instances
        assert self.num_instances > 0
        assert self.old_instances > 0

        self.obs_shape = (self.num_instances * 2 * self.team_size, get_obs_size(self.team_size))
        self.reward_shape = (self.num_instances * 2 * self.team_size)

        self.num_envs = (2 * self.num_instances - old_instances) * self.team_size

        self.mask_old_opponents = np.zeros(2 * self.num_instances * self.team_size, dtype=np.bool)

        # [[0], [1]]
        # [[0], [0], [1], [1]]
        # [[0], [0], [0], [1], [1], [1]]

        for i in range(self.old_instances):
            for j in range(self.team_size):
                self.mask_old_opponents[self.team_size * 2 * i + j] = True

        self.mask_new_opponent = np.logical_not(self.mask_old_opponents)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.work_queue = Queue()
        self.result_queue = Queue()
        p = Process(target=past_worker, args=(self.work_queue, self.result_queue, self.old_instances * self.team_size, device, url))
        p.start()

        start_past_model_downloader(url)

    def reset(self):
        obs = self.env.reset()

        obs_old = obs[self.mask_old_opponents]
        self.work_queue.put((obs_old, np.ones(self.old_instances * self.team_size, dtype=np.float32)))

        obs_new = obs[self.mask_new_opponent]

        return obs_new

    def step(self, action):
        combined_actions = np.empty([self.num_instances * 2 * self.team_size, 7], dtype=np.float32)
        combined_actions[self.mask_new_opponent] = action

        old_action = self.result_queue.get()
        combined_actions[self.mask_old_opponents] = old_action

        obs, rew, done, info = self.env.step(combined_actions)

        obs_old = obs[self.mask_old_opponents]
        done_old = done[self.mask_old_opponents]
        self.work_queue.put((obs_old, done_old))

        obs_new = obs[self.mask_new_opponent]
        rew_new = rew[self.mask_new_opponent]
        done_new = done[self.mask_new_opponent]

        return obs_new, rew_new, done_new, None

    def render(self, mode="human"):
        pass

    def get_monitor_data(self):
        return self.env.get_monitor_data()
