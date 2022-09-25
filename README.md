# Seer

I am training a reinforcement learning agent for Rocket League called __Seer__.

See [here](https://nevillewalo.ch/assets/docs/MA_Neville_Walo_Seer_RLRL.pdf), what Seer is about!

If you want to help train Seer, I can provide you with the necessary `host` and `port` to participate in the training!

## Requirements

* Windows PC
* Rocket League using Epic
* Bakkesmod
* Python [>=3.7,<=3.9]
* Git

## How to run

1. Open Rocket League and set it to potato quality (lowest resolution, windowed mode, lowest quality settings). Close Rocket Leagua again!
2. Start BakkesMod!
3. Open a PowerShell as administrator and put `Set-ExecutionPolicy RemoteSigned`
4. Clone this repository: `git clone  https://github.com/Walon1998/SeerWorker`
5. Enter directory: `cd SeerWorker`
6. Download and extract replays: `replay_downloader.ps1`
7. Run `worker.ps1 -N <N> --host <host> --port <port>`, where `N` is the number of Rocket League instances to launch
