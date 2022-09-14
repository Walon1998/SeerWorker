import argparse
import asyncio
import glob
import gzip
import io
import os
import pickle
import random
import struct
import time
from multiprocessing import Process

import aiohttp
import compress_pickle
import numpy as np
import torch
from SeerPPO import SeerNetwork, RolloutBuffer

from Env.MulitEnv import MultiEnv
from Env.NormalizeReward import NormalizeReward
from Env.PastEnv import download_models, past_model_downloader
from contants import GAMMA, HOST, PORT, LSTM_UNROLL_LENGTH, N_STEPS, GAE_LAMBDA


async def check_connection(session, url):
    async with session.get(url) as resp:
        assert resp.status == 200


async def get_reward_mean_var(session, url):
    async with session.get(url + "/mean_var") as resp:
        assert resp.status == 200
        bytes = await resp.read()
        mean = bytes[0:8]
        var = bytes[8:16]
        mean, var = struct.unpack('<d', mean), struct.unpack('<d', var)
        return mean[0], var[0]


async def put_reward_mean_var(session, url, mean, var):
    mean = struct.pack('<d', mean)
    var = struct.pack('<d', var)
    data = bytes(mean) + bytes(var)
    async with session.post(url + "/mean_var", data=data) as resp:
        assert resp.status == 200


async def update_policy(session, url, policy):
    async with session.get(url + "/Weights") as resp:
        assert resp.status == 200
        data, version = compress_pickle.loads(await resp.read(), compression="gzip")
        data = torch.load(io.BytesIO(data))
        policy.load_state_dict(data)
        policy.eval()
        return version


async def send_rollout(session, url, data, version):
    data = compress_pickle.dumps((data, version), compression="gzip")
    headers = {'Content-Encoding': 'gzip'}
    async with session.post(url + "/Data", data=data, headers=headers) as resp:
        assert resp.status == 200


async def collect_rollout_cpu(policy, env, buffer, init_data):
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
    buffer.compute_returns_and_advantage(last_values.numpy().ravel(), episode_starts)

    rollout = buffer.get_samples()
    buffer.reset()

    return rollout, (obs, lstm_states, episode_starts)


async def collect_rollout_gpu(policy, env, buffer, init_data):
    obs, lstm_states, episode_starts = init_data

    last_obs_cpu = obs.to("cpu", non_blocking=True)
    last_obs_gpu = obs.to("cuda", non_blocking=True)

    last_episode_starts_cpu = episode_starts.to("cpu", non_blocking=True)
    last_episode_starts_gpu = episode_starts.to("cuda", non_blocking=True)

    lstm_states_gpu = lstm_states[0].to("cuda", non_blocking=True), lstm_states[1].to("cuda", non_blocking=True)
    last_lstm_states_0_cpu = lstm_states[0].to("cpu", non_blocking=True)
    last_lstm_states_1_cpu = lstm_states[1].to("cpu", non_blocking=True)

    for _ in range(N_STEPS):
        torch.cuda.synchronize(device="cuda")

        with torch.no_grad():
            actions_gpu, values_gpu, log_probs_gpu, lstm_states_gpu = policy(last_obs_gpu, lstm_states_gpu, last_episode_starts_gpu, False)

        actions_cpu = actions_gpu.to("cpu", non_blocking=True)
        values_cpu = values_gpu.to("cpu", non_blocking=True)
        log_probs_cpu = log_probs_gpu.to("cpu", non_blocking=True)

        torch.cuda.synchronize(device="cuda")

        actions_cpu = actions_cpu.numpy()

        new_obs_cpu, rewards_cpu, episode_starts_cpu, infos = env.step(actions_cpu)

        new_obs_gpu = torch.as_tensor(new_obs_cpu, dtype=torch.float32).to("cuda", non_blocking=True)
        episode_starts_gpu = torch.as_tensor(episode_starts_cpu, dtype=torch.float32).to("cuda", non_blocking=True)
        lstm_states_0_cpu = lstm_states[0].to("cpu", non_blocking=True)
        lstm_states_1_cpu = lstm_states[1].to("cpu", non_blocking=True)

        buffer.add(last_obs_cpu.numpy(), actions_cpu, rewards_cpu, last_episode_starts_cpu.numpy(), values_cpu.numpy().ravel(), log_probs_cpu.numpy(), last_lstm_states_0_cpu.numpy(),
                   last_lstm_states_1_cpu.numpy())

        last_obs_gpu = new_obs_gpu
        last_obs_cpu = new_obs_cpu

        last_episode_starts_gpu = episode_starts_gpu
        last_episode_starts_cpu = episode_starts_cpu

        last_lstm_states_0_cpu = lstm_states_0_cpu
        last_lstm_states_1_cpu = lstm_states_1_cpu

    assert buffer.is_full()

    with torch.no_grad():
        last_values = policy.predict_value(new_obs_gpu, lstm_states, last_episode_starts_gpu)
    buffer.compute_returns_and_advantage(last_values.to("cpu").numpy().ravel(), last_episode_starts_cpu.numpy())

    rollout = buffer.get_samples()
    buffer.reset()

    return rollout, (last_obs_cpu, (last_lstm_states_0_cpu, last_lstm_states_1_cpu), last_episode_starts_cpu,)


async def RolloutWorker(args):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0, connect=60, sock_connect=60, sock_read=60)) as session:
        url = 'http://{host}:{port}'.format(host=HOST, port=PORT)
        print("Connecting to: ", url)
        await check_connection(session, url)
        print("Connection successful!")

        await download_models(session, url)
        p = Process(target=past_model_downloader, args=(url,))
        p.start()


        policy = SeerNetwork()
        policy.eval()

        mean, var = await get_reward_mean_var(session, url)
        env = MultiEnv(args["N"])
        env = NormalizeReward(env, mean, var, gamma=GAMMA)

        obs = env.reset()
        lstm_states = torch.zeros(1, args["N"] * 2, policy.LSTM.hidden_size, requires_grad=False), torch.zeros(1, args["N"] * 2, policy.LSTM.hidden_size,
                                                                                                               requires_grad=False)
        episode_starts = np.ones(args["N"] * 2)

        buffer = RolloutBuffer(N_STEPS, env.obs_shape[1], 7, args["N"] * 2, policy.LSTM.hidden_size, LSTM_UNROLL_LENGTH, GAMMA,
                               GAE_LAMBDA)

        init_data = obs, lstm_states, episode_starts

        version = await update_policy(session, url, policy)
        while True:
            start = time.time()

            if args["device"] == "cpu":
                rollout, init_data = await collect_rollout_cpu(policy, env, buffer, init_data)
            elif args["device"] == "cuda":
                rollout, init_data = await collect_rollout_cpu(policy, env, buffer, init_data)
            else:
                exit(-1)

            tasks = [asyncio.ensure_future(update_policy(session, url, policy)), asyncio.ensure_future(send_rollout(session, url, rollout, version))]
            if random.random() < 0.1:
                tasks.append(asyncio.ensure_future(put_reward_mean_var(session, url, env.return_rms.mean, env.return_rms.var)))
            res = await asyncio.gather(*tasks)
            version = res[0]

            end = time.time()

            fps = (N_STEPS * args["N"] * 2) / (end - start)
            print("FPS: {}".format(fps), end="\r")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', default=1, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--device_old', default="cpu", type=str)

    hyper_params = vars(parser.parse_args())

    asyncio.run(RolloutWorker(hyper_params))
