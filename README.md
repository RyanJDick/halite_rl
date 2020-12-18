## Getting Started

```
docker pull nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
docker build -t halite_rl .

# Note: The following settings are based on nvidia recommendation
# (https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html):
# --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864
docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ryan/src/halite_rl:/src/halite_rl -p 8888:8888 -p 6006:6006 halite_rl

# To run tensorboard from within the container (from /src/halite_rl)
tensorboard --logdir tensorboard_logs/ --bind_all

# To run jupyter from within container (from /src/halite_rl)
jupyter notebook --allow-root --port=8888 --no-browser --ip=0.0.0.0
```
## TODO:

* Further improve imitation bootstrapping.
* Set up self-play framework.
    * Read about how this was done in AlphaZero and other works.
    * Support both players learning at the same time? Or have one frozen?
    * Do I need to maintain a league of different agents? Or can I just have one agent playing against itself?
    * Single agent would probably be preferred given single GPU (and have to have a critic model on there as well)
* PPO
    * Implement simple non-parallel version to start.
    * Keep in mind eventual desire to have parallel environment updates:
        * Both agent being trained and opponent run NN to select actions. Actions are passed to environment, and environment step is applied. We then also want the conversion fro mhalite state to our preferred state representation to be parallelized.

* Have env that is rather inefficient, and provides state in a form that is not suitable for NNs.
    * Could re-implement the env to be faster, but I'd rather avoid this for now.
    * Key conversions: env_state_to_np_state_for_player, np_actions_to_board_actions_for_player
    * Create HaliteEnvWrapper for training. Can't use for actual gameplay, because they have their own runner. hence why the above utility functions should be kept separate so that they can also be used in a submission agent.
```

class Trainer: # analogous to ppo2 + runner in openai baselines (not sure if necessary to split)
    def __init__(self, players):
        # TBD should players share actor and critic networks?

    def train(self):

        # Create a bunch of environments
        # Roll them out in parallel, collecting minibatch examples in buffer.
```
