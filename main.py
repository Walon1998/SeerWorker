import argparse
import asyncio
import glob
import gzip
import io
import math
import os
import pickle
import random
import struct
import time
from multiprocessing import Process, Queue

import aiohttp
import compress_pickle
import numpy as np
import torch
from SeerPPO import RolloutBuffer
from SeerPPO.V2 import SeerNetworkV2
from SeerPPO.past_models_download import download_models
from shared_memory_dict import SharedMemoryDict

from Env.MonitorWrapper import MonitorWrapper
from Env.MulitEnv import MultiEnv
from Env.PastEnv import PastEnv
from contants import GAMMA, LSTM_UNROLL_LENGTH, N_STEPS, GAE_LAMBDA, PAST_MODELS


async def communication_worker_async(url, work_queue, result_queue):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0, connect=120, sock_connect=120, sock_read=120)) as session:
        print("Connecting to: ", url)
        await check_connection(session, url)
        print("Connection successful!")

        await download_models(session, url)

        state_dict, state_dict_version = await get_new_state_dict(session, url)

        result_queue.put((state_dict, state_dict_version))
        result_queue.put((state_dict, state_dict_version))

        while True:
            rollout, rollout_version = work_queue.get()

            task_high_priority = [asyncio.ensure_future(get_new_state_dict(session, url))]

            tasks_low_priority = [
                asyncio.ensure_future(send_rollout(session, url, rollout, rollout_version)),
            ]
            res = await asyncio.gather(*task_high_priority)

            result_queue.put(res[0])


def communication_worker(url, work_queue, result_queue):
    asyncio.run(communication_worker_async(url, work_queue, result_queue))


async def check_connection(session, url):
    async with session.get(url) as resp:
        assert resp.status == 200


async def get_new_state_dict(session, url):
    async with session.get(url + "/Weights") as resp:
        assert resp.status == 200
        data, version = compress_pickle.loads(await resp.read(), compression="gzip")
        data = torch.load(io.BytesIO(data), map_location="cpu")
        return data, version


def update_policy(policy, policy_version, state_dict, state_dict_version, device):
    if policy_version == state_dict_version:
        return policy_version

    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to(device)
    return state_dict_version


async def send_rollout(session, url, data, version):
    buffer, last_values, episode_starts, get_monitor_data = data

    buffer.compute_returns_and_advantage(last_values, episode_starts)
    data = buffer.get_samples(get_monitor_data)

    data = compress_pickle.dumps((data, version), compression="gzip")
    headers = {'Content-Encoding': 'gzip'}
    async with session.post(url + "/Data", data=data, headers=headers) as resp:
        assert resp.status == 200


def collect_rollout_cpu(policy, env, buffer, init_data):
    obs, lstm_states, episode_starts = init_data

    for _ in range(N_STEPS):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        episode_starts = torch.as_tensor(episode_starts, dtype=torch.float32)

        with torch.no_grad():
            actions, values, log_probs, new_lstm_states = policy(obs, lstm_states, episode_starts, False)

        actions = actions.numpy()

        new_obs, rewards, new_episode_starts, infos = env.step(actions)

        buffer.add(obs.numpy(), actions, rewards, episode_starts.numpy(), values.numpy().ravel(), log_probs.numpy(), lstm_states[0].numpy(), lstm_states[1].numpy())

        obs = new_obs
        episode_starts = new_episode_starts
        lstm_states = new_lstm_states

    assert buffer.is_full()
    with torch.no_grad():
        last_values = policy.predict_value(torch.as_tensor(obs, dtype=torch.float32), lstm_states, torch.as_tensor(episode_starts, dtype=torch.float32))

    rollout = buffer, last_values.numpy().ravel(), episode_starts, env.get_monitor_data()

    return rollout, (obs, lstm_states, episode_starts)


def collect_rollout_cuda(policy, env, buffer, init_data):
    last_obs, lstm_states, last_episode_start = init_data

    lstm_states = lstm_states[0].to("cuda", non_blocking=True), lstm_states[1].to("cuda", non_blocking=True)
    last_lstm_states_0_cpu = lstm_states[0].to("cpu", non_blocking=True)
    last_lstm_states_1_cpu = lstm_states[1].to("cpu", non_blocking=True)

    last_obs_gpu = torch.as_tensor(last_obs, dtype=torch.float32).to("cuda", non_blocking=True)

    last_episode_starts_gpu = torch.as_tensor(last_episode_start, dtype=torch.float32).to("cuda", non_blocking=True)

    for _ in range(N_STEPS):
        torch.cuda.synchronize(device="cuda")

        with torch.no_grad():
            action, value, log_probs, lstm_states = policy(last_obs_gpu, lstm_states, last_episode_starts_gpu, False)

        action = action.to("cpu", non_blocking=True)
        value = value.to("cpu", non_blocking=True)
        log_probs = log_probs.to("cpu", non_blocking=True)

        torch.cuda.synchronize(device="cuda")

        action = action.numpy()

        new_obs, rewards, dones, infos = env.step(action)

        new_obs_gpu = torch.as_tensor(new_obs, dtype=torch.float32).to("cuda", non_blocking=True)
        dones_gpu = torch.as_tensor(dones, dtype=torch.float32).to("cuda", non_blocking=True)
        lstm_states_0_cpu = lstm_states[0].to("cpu", non_blocking=True)
        lstm_states_1_cpu = lstm_states[1].to("cpu", non_blocking=True)
        buffer.add(last_obs, action, rewards, last_episode_start, value.numpy().ravel(), log_probs.numpy(), last_lstm_states_0_cpu.numpy(),
                   last_lstm_states_1_cpu.numpy())

        last_obs = new_obs
        last_obs_gpu = new_obs_gpu

        last_episode_start = dones
        last_episode_starts_gpu = dones_gpu

        last_lstm_states_0_cpu = lstm_states_0_cpu
        last_lstm_states_1_cpu = lstm_states_1_cpu

    assert buffer.is_full()

    torch.cuda.synchronize(device="cuda")

    with torch.no_grad():
        last_values = policy.predict_value(last_obs_gpu, lstm_states, last_episode_starts_gpu)

    rollout = buffer, last_values.to("cpu").numpy().ravel(), dones, env.get_monitor_data()

    return rollout, (last_obs, lstm_states, last_episode_start)


def RolloutWorker(args):
    url = 'http://{host}:{port}'.format(host=args["host"], port=args["port"])

    policy = SeerNetworkV2()
    policy.to(args["device"])
    policy.eval()
    smd_config = SharedMemoryDict(name='shared_memory_dict', size=1024)
    smd_config["step"] = 0

    work_queue = Queue()
    result_queue = Queue()

    p = Process(target=communication_worker, args=(url, work_queue, result_queue))
    p.start()

    env = MultiEnv(args["N"], args["team_size"], args["force_paging"])
    env = MonitorWrapper(env)

    if PAST_MODELS != 0.0:
        env = PastEnv(env, max(int(math.floor(args["N"] * PAST_MODELS)), 1), args["device_old"], url)

    obs = env.reset()
    lstm_states = torch.zeros(1, env.num_envs, policy.LSTM.hidden_size, requires_grad=False, dtype=torch.float32), torch.zeros(1, env.num_envs, policy.LSTM.hidden_size,
                                                                                                                               requires_grad=False, dtype=torch.float32)
    episode_starts = np.ones(env.num_envs, dtype=np.float32)

    init_data = obs, lstm_states, episode_starts

    policy_version = -1

    while True:
        start = time.time()

        buffer = RolloutBuffer(N_STEPS, env.obs_shape[1], 7, env.num_envs, policy.LSTM.hidden_size, LSTM_UNROLL_LENGTH, GAMMA,
                               GAE_LAMBDA)

        state_dict, state_dict_version = result_queue.get()
        policy_version = update_policy(policy, policy_version, state_dict, state_dict_version, args["device"])
        smd_config["step"] = policy_version

        if args["device"] == "cpu":
            rollout, init_data = collect_rollout_cpu(policy, env, buffer, init_data)
        elif args["device"] == "cuda":
            rollout, init_data = collect_rollout_cuda(policy, env, buffer, init_data)
        else:
            exit(-1)

        work_queue.put((rollout, policy_version))

        end = time.time()

        fps = (N_STEPS * env.num_envs) / (end - start)
        print("FPS: {}".format(fps), end="\r")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', default=1, type=int)
    parser.add_argument('--team_size', type=int, required=True)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--device_old', default="cuda", type=str)
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--force_paging', default=True, type=bool)

    hyper_params = vars(parser.parse_args())

    RolloutWorker(hyper_params)
