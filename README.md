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
2. Open PowerShell in your preferred directory
3. Clone this repository: `git clone  https://github.com/Walon1998/SeerWorker`
4. Enter directory: `cd SeerWorker`
5. Extract replays: `cd Replays;$files = "Replays_0.zip", "Replays_1.zip","Replays_2.zip","Replays_3.zip","Replays_4.zip";foreach ($f in $files){  Expand-Archive -DestinationPath ".\" $f};cd ..`
6. Run `worker.ps1 -N <N> --host <host> --port <port>`, where `N` is the number of Rocket League instances to launch
    * You might have to set the PowerShell Execution Policy from Restricted to RemoteSigned to allow local PowerShell scripts to run. Open a PowerShell as administrator and put `Set-ExecutionPolicy RemoteSigned`.