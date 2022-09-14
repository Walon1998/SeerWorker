import argparse
import asyncio
import gzip
import io
import time

import aiohttp
import compress_pickle
import numpy as np
import torch
from SeerPPO import SeerNetwork, RolloutBuffer
from gym.wrappers import NormalizeReward

from Env.MulitEnv import MultiEnv
from contants import GAMMA, HOST, PORT, LSTM_UNROLL_LENGTH, N_STEPS, GAE_LAMBDA


async def check_connection(session, url):
    async with session.get(url) as resp:
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


async def RolloutWorker(args):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0, connect=60, sock_connect=60, sock_read=60)) as session:
        url = 'http://{host}:{port}'.format(host=HOST, port=PORT)
        print("Connecting to: ", url)
        await check_connection(session, url)
        print("Connection successful!")

        policy = SeerNetwork()
        policy.eval()

        env = MultiEnv(args["N"])
        env = NormalizeReward(env, gamma=GAMMA)

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

            rollout, init_data = await collect_rollout_cpu(policy, env, buffer, init_data)
            tasks = [asyncio.ensure_future(update_policy(session, url, policy)), asyncio.ensure_future(send_rollout(session, url, rollout, version))]
            res = await asyncio.gather(*tasks)
            version = res[0]

            end = time.time()

            fps = (N_STEPS * args["N"] * 2) / (end - start)
            print("FPS: {}".format(fps), end="\r")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', default=1, type=int)
    parser.add_argument('--device', default="cpu", type=str)

    hyper_params = vars(parser.parse_args())

    asyncio.run(RolloutWorker(hyper_params))
