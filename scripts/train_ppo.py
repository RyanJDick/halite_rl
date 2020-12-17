import argparse
from datetime import datetime
import yaml

import numpy as np
import torch

from halite_rl.utils import (
    HaliteActorCriticCNN,
)
from halite_rl.ppo.sample import sample_batch
from halite_rl.ppo.halite_env_wrapper import HaliteEnvWrapper

class BatchData():
    def __init__(self):
        self.states = []              # States
        self.actions = []             # Selected actions.
        self.fixed_act_log_probs = [] # Log probability of selected action at time of execution. (frozen weights, no gradient flow).
        self.returns = []             # Sum of discounted future rewards.
        self.advantages = []          # Advantage of action taken over default policy behaviour.


def estimate_advantages(rewards, value_preds, gamma, lambda_, normalize_advantages):
    """Estimate advantages / returns for episode rollout according to 
    Generalized Advantage Estimation (GAE).
    """

    deltas = np.zeros(len(rewards))
    advantages = np.zeros(len(rewards))

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(len(rewards))):
        # Error between critic-predicted value and actual discounted return.
        # This can be considered an estimate of the advantage of action at time i.
        deltas[i] = rewards[i] + gamma * prev_value - value_preds[i]

        # See paper for derivation of this advantage estimate.
        # Increasing lambda_ from 0 to 1 generally leads to increasing variance,
        # but decreasing bias in the advantage estimate.
        advantages[i] = deltas[i] + gamma * lambda_ * prev_advantage

        prev_value = value_preds[i]
        prev_advantage = advantages[i]

    returns = value_preds + advantages
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages, returns

def ep_rollouts_to_training_batch(ep_rollouts, gamma, lambda_, normalize_advantages):
    """Process episode rollouts according to Generalized Advantage Estimation (GAE)
    to extract training examples.
    """
    bd = BatchData()
    states = []
    actions = []
    returns = []
    advantages = []
    fixed_act_log_probs = []

    for ep_data in ep_rollouts:
        ep_rewards = np.array(ep_data.rewards)
        ep_value_preds = np.array(ep_data.value_preds)
        ep_advantages, ep_returns = estimate_advantages(ep_rewards, ep_value_preds, gamma, lambda_, normalize_advantages)

        states.append(np.array(ep_data.observations))
        actions.append(np.array(ep_data.actions))
        returns.append(np.array(ep_returns))
        advantages.append(np.array(ep_advantages))
        fixed_act_log_probs.append(np.array(ep_data.act_log_probs))

    bd.states = np.concatenate(states)
    bd.actions = np.concatenate(actions)
    bd.returns = np.concatenate(returns)
    bd.advantages = np.concatenate(advantages)
    bd.fixed_act_log_probs = np.concatenate(fixed_act_log_probs)
    return bd

def ppo_param_update(model, optimizer, batch_data, config, device):
    """Perform mini-batch PPO update."""
    batch_size = len(batch_data.returns)
    num_batch_update_epochs = 4 # Number of times to perform gradient updates on this batch. TODO: add to config

    for batch_epoch in range(num_batch_update_epochs):
        print(f"Start of batch update epoch {batch_epoch}.")

        # Process examples in different order in each batch epoch.
        perm = np.arange(batch_size)
        np.random.shuffle(perm)

        for mb_start in range(0, batch_size, config["MINIBATCH_SIZE"]):
            print(f"Training on minibatch {mb_start} to {mb_start + config['MINIBATCH_SIZE']}")
            idxs = perm[mb_start:mb_start+config["MINIBATCH_SIZE"]]
            states = torch.from_numpy(batch_data.states[idxs, ...]).to(device)
            actions = torch.from_numpy(batch_data.actions[idxs, ...]).to(device)
            ship_actions = actions[:, 0, :, :]
            shipyard_actions = actions[:, 1, :, :]
            fixed_act_log_probs = torch.from_numpy(batch_data.fixed_act_log_probs[idxs, ...]).to(device)
            returns = torch.from_numpy(batch_data.returns[idxs, ...]).to(device)
            advantages = torch.from_numpy(batch_data.advantages[idxs, ...]).to(device)

            print(f"Returns has shape: {returns.shape}")
            print(f"ship_actions has shape: {ship_actions.shape}")

            action_logits, value_preds, _, _, _ = model(states)
            ship_action_logits = action_logits[:, :config["NUM_SHIP_ACTIONS"], :, :]
            shipyard_action_logits = action_logits[:, config["NUM_SHIP_ACTIONS"]:, :, :]

            ship_action_dist = model.apply_action_distribution(ship_action_logits)
            shipyard_action_dist = model.apply_action_distribution(shipyard_action_logits)

            # TODO: I like this better: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py#L80-L83

            # TODO: ignore empty locations?
            # TODO: consolidate with duplicate code in sampler.py Perhaps this belongs in the model itself?
            ship_action_log_probs = ship_action_dist.log_prob(ship_actions)
            shipyard_action_log_probs = shipyard_action_dist.log_prob(shipyard_actions)
            action_log_probs = ship_action_log_probs + shipyard_action_log_probs

            value_loss = (value_preds - returns).pow(2).mean()

            # Actor loss.
            # TODO: review this block (copied from elsewhere)
            ratio = torch.exp(action_log_probs - fixed_act_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - config["PPO_CLIP_EPSILON"], 1.0 + config["PPO_CLIP_EPSILON"])
            policy_loss = -torch.min(surr1, surr2).mean()

            total_loss = value_loss * config["VALUE_LOSS_COEFF"] + policy_loss
            # TODO: add L2 regularization like here?: https://github.com/Khrylx/PyTorch-RL/blob/d94e1479403b2b918294d6e9b0dc7869e893aa1b/core/ppo.py#L12
            # TODO: add policy distribution entropy term like here?: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py#L81
            optimizer.zero_grad()
            total_loss.backward()
            # TODO: nn.utils.clip_grad_norm_(model.parameters(), config["MAX_GRAD_CLIP_NORM"])
            optimizer.step()


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
    train_player_id = 0

    for player_id, ckpt_path in [
        (train_player_id, config["TRAIN_MODEL_CHECKPOINT_PATH"]),
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
    train_model = models[train_player_id]

    # 5. Initialize optimizer.
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.00001)
    if config["LOAD_TRAIN_OPTIMIZER_FROM_CHECKPOINT"]:
        checkpoint = torch.load(config["TRAIN_MODEL_CHECKPOINT_PATH"], map_location=dev)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 6. Train.
    for epoch in range(10000):
        # Sample episode rollouts.
        ep_rollouts = sample_batch(models, HaliteEnvWrapper, dev, config)
        print("received batch")
        print(f"batch contains: {len(ep_rollouts[train_player_id])} episodes")

        # Compute returns/advantages.
        batch_data = ep_rollouts_to_training_batch(
            ep_rollouts[train_player_id], config["GAE_GAMMA"], config["GAE_LAMBDA"], config["NORMALIZE_ADVANTAGES"])
        print("Computed returns/advantages:")
        print(f"states.shape: {batch_data.states.shape}")
        print(f"actions.shape: {batch_data.actions.shape}")
        print(f"returns.shape: {batch_data.returns.shape}")
        print(f"advantages.shape: {batch_data.advantages.shape}")
        print(f"fixed_act_log_probs.shape: {batch_data.fixed_act_log_probs.shape}")
        print(f"returns[:10]: {batch_data.returns[:10]}")
        print(f"advantages[:10]: {batch_data.advantages[:10]}")

        # Perform parameters updates.
        ppo_param_update(train_model, optimizer, batch_data, config, dev)





# Before thinking about league dynamics, need a way to train 1 agent against another (frozen agent).

# Configs:
# - num steps to run
# - train batch size
# - num_iters (times to go back and forth between sampling and training)