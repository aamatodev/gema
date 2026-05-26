# Graph Embeddings for Multi-Agent Coordination

Official repository for **“Encoding Goals as Graphs: Structured Objectives for Scalable Cooperative Multi-Agent Reinforcement Learning.”**

## How to use it

### 1. Docker (recommended)

Using the Dockerfile is the fastest and most convenient way to set up the environment.
These steps were tested on a Linux machine running Ubuntu 22.04 with an NVIDIA H100 and CUDA 12.8.

First, clone the repository:

```bash
git clone https://github.com/aamatodev/gema.git
```

Then, build the Docker image from the root directory of the repository:

```bash
docker build -t aamatodev/gema .
```

If you have trouble building the container, you may have a CUDA/hardware incompatibility. Make sure the `FROM` line in the Dockerfile matches your CUDA version. If you change it, also update the CUDA-related settings around line ~20 in the Dockerfile accordingly.
As of **January 28, 2026**, **torch_geometric** supports **PyTorch up to 2.8.0**.

If you can’t get Docker working, please open an issue.

To run the Docker container, use:

```bash
docker run --gpus device=0 -it --ipc=host --name=aamatodev_gema_gpu0 aamatodev/gema
```

`--gpus device=0` selects the GPU to use. Adjust it as needed.

To train **load balancing**, run:

```bash
xvfb-run -a python3 train_gema_balancing.py
```

To train **cooperative navigation**, run:

```bash
xvfb-run -a python3 train_gema_spread.py
```

You will be prompted to choose whether to log the run to Weights & Biases (wandb).

## Official papers

* Amato, Alessandro, et al. *“Encoding Goals as Graphs: Structured Objectives for Scalable Cooperative Multi-Agent Reinforcement Learning.”* The 25th International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS), 2026.
* Amato, Alessandro, et al. *“Encoding Goals as Graphs: Structured Objectives for Scalable Cooperative Multi-Agent Reinforcement Learning.”* Second Coordination and Cooperation in Multi-Agent Reinforcement Learning Workshop, 2025.

## Citation

```bibtex
@inproceedings{
  amato2026encoding,
  title={Encoding Goals as Graphs: Structured Objectives for Scalable Cooperative Multi-Agent Reinforcement Learning},
  author={Alessandro Amato and Raffaele Galliera and K. Brent Venable and Niranjan Suri},
  booktitle={The 25th International Conference on Autonomous Agents and Multi-Agent Systems},
  year={2026}
}
```
