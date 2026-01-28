# Graph Embedding for Multi-Agent Coordination
Official Repository for "Encoding Goals as Graphs: Structured Objectives for Scalable Cooperative MultiAgent Reinforcement Learning"


## How to use it
### 1. Docker (Recommended)
Using the Dockerfile is the fastest and most convenient way to set up the environment.
These steps where tested on a Linux machine running Ubuntu 22.04, an Nvidia H100 with CUDA 12.8 .

First, clone the repository:
```bash
git clone https://github.com/aamatodev/gema.git
```

Then, to build the docker image run the following command in the root directory of the repository:
```bash
docker build -t aamatodev/gema .
```
If you have troubles building the container, you'll probably have some incompatibility with the CUDA version and your hardware. Make sure to adjust the `FROM` line in the Dockerfile to match your CUDA version. If you change do so,
update like 20 of the docker file to match your new cuda version. As of Jan 28 2026, torch_geometric support torch up to 2.8.0.
If you can't get the docker to work, open an issue.

To run the docker container, use the following command:
```bash
docker run --gpus device=0 -it --ipc=host --name=aamatodev_gema_gpu0 aamatodev/gema
```
`--gpus device=0` specifies which GPU to use. Adjust it as needed.

To train load_balancing use:
```bash
xvfb-run -a python3 train_gema_balancing.py
```


To train cooperative navigation use:
```bash
xvfb-run -a python3 train_gema_spread.py
```

You'll propmt to choose wheter to save the run on wandb or not.

## Official Papers:
- Amato, Alessandro, et al. "Encoding Goals as Graphs: Structured Objectives for Scalable Cooperative Multi-Agent Reinforcement Learning." The 25th International Conference on Autonomous Agents and Multi-Agent Systems, 2026.
- Amato, Alessandro, et al. "Encoding goals as graphs: Structured objectives for scalable cooperative multi-agent reinforcement learning." Second Coordination and Cooperation in Multi-Agent Reinforcement Learning Workshop. 2025.

## Citation
```shell
@inproceedings{
amato2026encoding,
title={Encoding Goals as Graphs: Structured Objectives for Scalable Cooperative Multi-Agent Reinforcement Learning},
author={Alessandro Amato and Raffaele Galliera and K. Brent Venable and Niranjan Suri},
booktitle={The 25th International Conference on Autonomous Agents and Multi-Agent Systems},
year={2026},
url={https://openreview.net/forum?id=R0La9WDfgw}
}
```