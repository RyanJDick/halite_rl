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

## TODO:

* Current priority: settle on network arch that gives good results for both action and value prediction.
* Train PPO from imitation model.
* improve value prediction by including more skip connections from inputs to final layers
* look into gradient clipping
* Shoudn't be using root user in docker container
* Profiling to find bottlenecks and speed up training. I am under the impression that the halite environment implementation is really slow.
* Set up self-play framework.
    * Read about how this was done in AlphaZero and other works.
    * Support both players learning at the same time? Or have one frozen?
    * Do I need to maintain a league of different agents? Or can I just have one agent playing against itself?
    * Single agent would probably be preferred given single GPU (and have to have a critic model on there as well)
