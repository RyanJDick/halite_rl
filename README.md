# halite_rl

This repo contains an experimental RL project based on the Halite game used in [this](https://www.kaggle.com/c/halite-iv-playground-edition) Kaggle challenge. The challenge was nearly complete by the time that I stumbled upon it, but it nonetheless caught my attention as a potentially interesting reinforcement learning environment. The goal of this project is to train an RL agent to master the game of Halite. As I get deeper into this project, I'll tidy up the documentation here and share my results.

## Getting Started

```
docker pull nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
docker build -t halite_rl .

# Note: The following settings are based on nvidia recommendation
# (https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html):
# --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864
docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/src/halite_rl -p 8888:8888 -p 6006:6006 halite_rl

# To run tensorboard from within the container (from /src/halite_rl)
tensorboard --logdir tensorboard_logs/ --bind_all

# To run jupyter from within container (from /src/halite_rl)
jupyter notebook --allow-root --port=8888 --no-browser --ip=0.0.0.0
```

See [imitation README](halite_rl/imitation/README.md) and [PPO README](halite_rl/ppo/README.md) for next steps.

## Unit Tests
```
pip install -e .
pytest tests/
```
