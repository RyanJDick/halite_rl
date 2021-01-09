# halite_rl

This repo contains an experimental RL project based on the Halite game used in [this](https://www.kaggle.com/c/halite-iv-playground-edition) Kaggle challenge. The challenge was nearly complete by the time that I stumbled upon it, but it nonetheless caught my attention as a potentially interesting reinforcement learning environment. The goal of this project is to train an RL agent to master the game of Halite. As I get deeper into this project, I'll tidy up the documentation here and share my results.

## Approach

The goal of this project is to train an RL agent to play the game of Halite, with the intention to take away some practical learnings about RL. The goal is **not** to build the best possible Halite agent.

### Imitation Learning

As a first step, I decided to try and train an agent to imitate expert behaviour. The "expert" in this case being the behaviour of the top-performing submission on the Kaggle leaderboard. I had two main justifications for starting with imitation learning:
1. To evaluate some of the key model decisions in a simplified environment before trying to apply the model in a much less stable RL training algorithm. My thinking here was that if I were succesfully able to train a model to imitate (to a reasonable level) a hand-engineered expert policy then that would give me confidence that my model is 1) capable of extracting the key features used by a human in designing the expert policy, and 2) sufficiently expressive to make reasonable decisions based on these features.
2. In order to bootstrap the RL training process. My hope is that by starting with a pretrained imitation model I will be able to accelerate the RL training process.

Results to come...

### PPO

...

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
