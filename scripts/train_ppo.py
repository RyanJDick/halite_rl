import argparse
from datetime import datetime
import yaml

import torch

from halite_rl.utils import (
    HaliteActorCriticCNN,
)
from halite_rl.ppo.sample import sample_batch
from halite_rl.ppo.halite_env_wrapper import HaliteEnvWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()

    # 0. Load configs.
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    # 2. Select device.
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    dev = torch.device(dev)

    # 3. Assemble model name.
    timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    train_model_name = f"{config['MODEL_NAME']}_{timestamp}"

    # 4. Load checkpoints.
    models = {}

    for player_id, ckpt_path in [
        (0, config["TRAIN_MODEL_CHECKPOINT_PATH"]),
        (1, config["OPPONENT_MODEL_CHECKPOINT_PATH"]),
    ]:
        checkpoint = torch.load(ckpt_path, map_location=dev)
        net = HaliteActorCriticCNN(
            num_actions=config["NUM_SHIP_ACTIONS"] + config["NUM_SHIPYARD_ACTIONS"],
            input_hw=config["BOARD_HW"],
        )
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(dev)
        models[player_id] = net
        print(f"Loaded player '{player_id}' model from: '{ckpt_path}'")
    train_model = models[0]

    # 5. Train.
    for epoch in range(10000):
        # Sample episode rollouts.
        batch = sample_batch(models, HaliteEnvWrapper, dev, config)
        print("received batch")
        print(f"batch contains: {len(batch[0])} episodes")
        ep = batch[0][0]
        print(f"len(ep.observations): {len(ep.observations)}")
        print(f"len(ep.rewards): {len(ep.rewards)}")
        print(f"ep.rewards[-1]: {ep.rewards[-1]}")
        break




# Before thinking about league dynamics, need a way to train 1 agent against another (frozen agent).

# Configs:
# - num steps to run
# - train batch size
# - num_iters (times to go back and forth between sampling and training)