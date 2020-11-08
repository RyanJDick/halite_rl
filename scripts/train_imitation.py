import argparse
from datetime import datetime
import os
import json

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-id", help="submission_id for which to download episode")
    parser.add_argument("--team-name", help="TeamName associated with submission-id")
    args = parser.parse_args()

    # 1. Initialize dataset.
    episodes_dir = f"./data/submission_{args.submission_id}_episodes/"
    episode_files = sorted(os.listdir(episodes_dir))
    episode_files = [os.path.join(episodes_dir, f) for f in episode_files]

    dataset = HaliteStateActionDataset(episode_files, args.team_name)

    loader = DataLoader(dataset, batch_size=4, num_workers=0, worker_init_fn=hsap_worker_init_fn)

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
