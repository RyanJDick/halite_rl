import argparse
from collections import defaultdict
from datetime import datetime
import os
import yaml

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from halite_rl.utils import (
    HaliteActorCriticCNN,
    SHIP_ACTION_ID_TO_NAME,
    SHIPYARD_ACTION_ID_TO_NAME,
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

    losses = defaultdict(list)
    loss_minibatch_sizes = [] # for computing weighted mean if minibatch sizes are not always the same.

    for batch_epoch in range(config["BATCH_UPDATE_EPOCHS"]):
        # Process examples in different order in each batch epoch.
        perm = np.arange(batch_size)
        np.random.shuffle(perm)

        for mb_start in range(0, batch_size, config["MINIBATCH_SIZE"]):
            idxs = perm[mb_start:mb_start+config["MINIBATCH_SIZE"]]
            states = torch.from_numpy(batch_data.states[idxs, ...]).to(device)
            actions = torch.from_numpy(batch_data.actions[idxs, ...]).to(device)
            ship_actions = actions[:, 0, :, :]
            shipyard_actions = actions[:, 1, :, :]
            fixed_act_log_probs = torch.from_numpy(batch_data.fixed_act_log_probs[idxs, ...]).to(device)
            returns = torch.from_numpy(batch_data.returns[idxs, ...]).to(device)
            advantages = torch.from_numpy(batch_data.advantages[idxs, ...]).to(device)

            ship_act_logits, shipyard_act_logits, value_preds = model(states)

            ship_action_dist, shipyard_action_dist = model.get_action_distribution(
                    ship_act_logits, shipyard_act_logits, states)

            action_log_probs = model.action_log_prob(
                ship_action_dist,
                shipyard_action_dist,
                ship_actions,
                shipyard_actions,
            )

            value_loss = (value_preds - returns).pow(2).mean()

            # Actor loss.
            prob_ratio = torch.exp(action_log_probs - fixed_act_log_probs)
            surr_obj_1 = prob_ratio * advantages
            eps = config["PPO_CLIP_EPSILON"]
            surr_obj_2 = torch.clamp(prob_ratio, 1.0 - eps, 1.0 + eps) * advantages

            # Note: surr_obj_1 and surr_obj_2 are equivalent before any parameter updates have been applied,
            # but they will drift apart as the policy network parameters change from the old values
            # (that the data was sampled with).
            # Multiply by -1 to transform from objective to loss.
            policy_loss = -torch.min(surr_obj_1, surr_obj_2).mean()

            total_loss = value_loss * config["VALUE_LOSS_COEFF"] + policy_loss
            # TODO see all losses used here: https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#L115-L116
            # TODO: add L2 regularization like here?: https://github.com/Khrylx/PyTorch-RL/blob/d94e1479403b2b918294d6e9b0dc7869e893aa1b/core/ppo.py#L12
            # TODO: add policy distribution entropy term like here?: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py#L81
            optimizer.zero_grad()
            total_loss.backward()
            # TODO: nn.utils.clip_grad_norm_(model.parameters(), config["MAX_GRAD_CLIP_NORM"])
            optimizer.step()

            losses["value_mse"].append(value_loss.item())
            losses["policy_loss"].append(policy_loss.item())
            loss_minibatch_sizes.append(len(idxs))

    # Calculate weighted (by minibatch size) mean losses.
    # TODO: a little weird that we're aggregating losses from multiple updates on the same examples.
    mean_losses = {}
    loss_minibatch_sizes = np.array(loss_minibatch_sizes)
    for loss_name, loss_list in losses.items():
        mean_losses[loss_name] = (np.array(loss_list) * loss_minibatch_sizes).sum() / loss_minibatch_sizes.sum()

    return mean_losses


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
        net = HaliteActorCriticCNN(
            input_hw=config["BOARD_HW"],
            num_ship_actions=config["NUM_SHIP_ACTIONS"],
            num_shipyard_actions=config["NUM_SHIPYARD_ACTIONS"],
        )

        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=dev)
            net.load_state_dict(checkpoint['model_state_dict'])

        net.to(dev)
        models[player_id] = net
        print(f"Loaded player '{player_id}' model from: '{ckpt_path}'")
    train_model = models[train_player_id]

    # 5. Initialize optimizer.
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)
    if config["LOAD_TRAIN_OPTIMIZER_FROM_CHECKPOINT"]:
        checkpoint = torch.load(config["TRAIN_MODEL_CHECKPOINT_PATH"], map_location=dev)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 6. Initialize tensorboard writer.
    tensorboard_base_dir = './tensorboard_logs_ppo/'
    tensorboard_dir = os.path.join(tensorboard_base_dir, train_model_name)
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    print("Enter a description for this run...")
    run_description = input()
    tensorboard_writer.add_text("description", run_description)
    print("Starting training...")

    # 6. Train.
    ep_tot_returns = []
    ep_lens = []
    ep_ship_action_dist_entropys = []
    ep_shipyard_action_dist_entropys = []
    running_ship_action_counts = np.zeros(config["NUM_SHIP_ACTIONS"])
    running_shipyard_action_counts = np.zeros(config["NUM_SHIPYARD_ACTIONS"])
    report_epoch_freq = 5
    for epoch in range(10000):
        print(f"epoch {epoch}")

        # Sample episode rollouts.
        ep_rollouts = sample_batch(models, HaliteEnvWrapper, dev, config)

        # Record episode metrics.
        for ep_data in ep_rollouts[train_player_id]:
            ep_tot_returns.append(np.sum(ep_data.rewards))
            ep_lens.append(len(ep_data.rewards))
            for si in ep_data.step_info:
                ep_ship_action_dist_entropys.append(si["ship_action_dist_entropy"])
                ep_shipyard_action_dist_entropys.append(si["shipyard_action_dist_entropy"])
                running_ship_action_counts += si["ship_action_counts"]
                running_shipyard_action_counts += si["shipyard_action_counts"]

        # Compute returns/advantages.
        batch_data = ep_rollouts_to_training_batch(
            ep_rollouts[train_player_id], config["GAE_GAMMA"], config["GAE_LAMBDA"], config["NORMALIZE_ADVANTAGES"])

        # Perform parameters updates.
        mean_losses = ppo_param_update(train_model, optimizer, batch_data, config, dev)

        # Report losses to tensorboard.
        for name, loss in mean_losses.items():
            tensorboard_writer.add_scalar(f'Loss/{name}', loss, epoch+1)

        # Report episode statistics to tensorboard and log.
        if (epoch + 1) % report_epoch_freq == 0:
            mean_tot_return = np.mean(ep_tot_returns)
            mean_ep_len = np.mean(ep_lens)

            print(f"mean_tot_return: {mean_tot_return}, mean_ep_len: {mean_ep_len}")
            mean_ship_action_dist_entropy = np.mean(ep_ship_action_dist_entropys)
            mean_shipyard_action_dist_entropy = np.mean(ep_shipyard_action_dist_entropys)
            print(f"mean_ship_action_dist_entropy: {mean_ship_action_dist_entropy}, "
                f"mean_shipyard_action_dist_entropy: {mean_shipyard_action_dist_entropy}")
            print("Ship Action Counts:")
            tot_ship_actions = running_ship_action_counts.sum()
            for act_i in range(config["NUM_SHIP_ACTIONS"]):
                print(f"\t({act_i}) {SHIP_ACTION_ID_TO_NAME[act_i]}: {running_ship_action_counts[act_i]} "
                    f"({running_ship_action_counts[act_i] / tot_ship_actions:.4f})")
            print("Shipyard Action Counts:")
            tot_ship_actions = running_shipyard_action_counts.sum()
            for act_i in range(config["NUM_SHIPYARD_ACTIONS"]):
                print(f"\t({act_i}) {SHIPYARD_ACTION_ID_TO_NAME[act_i]}: {running_shipyard_action_counts[act_i]} "
                    f"({running_shipyard_action_counts[act_i] / tot_ship_actions:.4f})")

            tensorboard_writer.add_scalar(f'EpisodeStats/mean_tot_return', mean_tot_return, epoch+1)
            tensorboard_writer.add_scalar(f'EpisodeStats/mean_ep_len', mean_ep_len, epoch+1)
            tensorboard_writer.add_scalar(f"EpisodeStats/mean_ship_action_dist_entropy", mean_ship_action_dist_entropy, epoch+1)
            tensorboard_writer.add_scalar(f"EpisodeStats/mean_shipyard_action_dist_entropy", mean_ship_action_dist_entropy, epoch+1)
            tensorboard_writer.add_histogram(f"EpisodeStats/ship_action_counts", running_ship_action_counts, epoch+1)
            tensorboard_writer.add_histogram(f"EpisodeStats/shipyard_action_counts", running_shipyard_action_counts, epoch+1)

            # Reset all running stats.
            ep_tot_returns = []
            ep_lens = []
            ep_ship_action_dist_entropys = []
            ep_shipyard_action_dist_entropys = []
            running_ship_action_counts = np.zeros(config["NUM_SHIP_ACTIONS"])
            running_shipyard_action_counts = np.zeros(config["NUM_SHIPYARD_ACTIONS"])
