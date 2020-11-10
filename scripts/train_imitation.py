import argparse
from datetime import datetime
import os
import json
import yaml

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn

from halite_rl.imitation import (
    HaliteStateActionDataset,
    hsap_worker_init_fn,
    ImitationCNN,
)

def load_episode_list_from_file(base_dir, episodes_file):
    with open(os.path.join(base_dir, episodes_file)) as f:
        episodes = f.readlines()
    episodes = [os.path.join(base_dir, f.strip()) for f in episodes]
    return sorted(episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()

    # 0. Load configs.
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    # 1. Initialize dataset.
    train_episode_files = load_episode_list_from_file(config["BASE_DATA_DIR"], config["TRAIN_EPISODES_FILE"])
    train_dataset = HaliteStateActionDataset(train_episode_files, config["TEAM_NAME"])
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=0, worker_init_fn=hsap_worker_init_fn)

    val_episode_files = load_episode_list_from_file(config["BASE_DATA_DIR"], config["VAL_EPISODES_FILE"])
    val_dataset = HaliteStateActionDataset(val_episode_files, config["TEAM_NAME"])
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=0, worker_init_fn=hsap_worker_init_fn)

    # 2. Initialize network.
    net = ImitationCNN()

    # 3. Define loss function / optimizer.
    ship_action_ce = nn.CrossEntropyLoss()
    shipyard_action_ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 4. Train the network
    tensorboard_base_dir = './tensorboard_logs/'
    timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    tensorboard_dir = os.path.join(tensorboard_base_dir, f"model-name_{timestamp}")
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    for epoch in range(10):
        running_loss = 0.0
        for i, batch in enumerate(loader):
            state, ship_actions, shipyard_actions = batch

            # zero the parameter gradients.
            optimizer.zero_grad()

            # forward + backward + optimize.
            outputs = net(state)
            ship_action_loss = ship_action_ce(outputs[:, :6, :, :], ship_actions)
            shipyard_action_loss = shipyard_action_ce(outputs[:, 6:, :, :], shipyard_actions)
            loss = ship_action_loss + shipyard_action_loss
            loss.backward()
            optimizer.step()

            # print statistics.
            running_loss += loss.item()
            stats_freq = 1000
            if i % stats_freq == stats_freq-1:
                # Don't log first set of batches to tensorboard, as the visualizations
                # don't scale well when the first point is an outlier.
                if i > stats_freq:
                    tensorboard_writer.add_scalar('Loss/train', running_loss / stats_freq, i)
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / stats_freq:.5f}")
                running_loss = 0.0
