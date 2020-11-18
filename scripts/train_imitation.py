import argparse
from collections import defaultdict
from datetime import datetime
import os
import time

import h5py
import json
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn

from halite_rl.imitation import (
    ImitationCNN,
    HaliteStateActionHDF5Dataset,
)
from halite_rl.utils import (
    SHIP_ACTION_ID_TO_NAME,
    SHIPYARD_ACTION_ID_TO_NAME,
)

def load_episode_list_from_file(base_dir, episodes_file):
    with open(os.path.join(base_dir, episodes_file)) as f:
        episodes = f.readlines()
    episodes = [os.path.join(base_dir, f.strip()) for f in episodes]
    return sorted(episodes)

def update_running_confusion_matrix(
    outputs,
    actions_gt,
    confusion_matrix,
):
    num_classes = outputs.shape[1]
    preds = np.argmax(outputs, axis=1) # outputs NCHW -> NHW

    for gt_class in range(num_classes):
        for pred_class in range(num_classes):
            confusion_matrix[gt_class, pred_class] += \
                np.count_nonzero((actions_gt == gt_class) & (preds == pred_class))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()

    # 0. Load configs.
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    # 1. Initialize dataset.
    train_dataset_file = os.path.join(config["BASE_DATA_DIR"], config["TRAIN_HDF5_FILE"])
    train_data_hdf5 = h5py.File(train_dataset_file, 'r')
    train_dataset = HaliteStateActionHDF5Dataset(train_data_hdf5)
    train_loader = DataLoader(train_dataset, batch_size=1000, num_workers=8)

    val_dataset_file = os.path.join(config["BASE_DATA_DIR"], config["VAL_HDF5_FILE"])
    val_data_hdf5 = h5py.File(val_dataset_file, 'r')
    val_dataset = HaliteStateActionHDF5Dataset(val_data_hdf5)
    val_loader = DataLoader(val_dataset, batch_size=1000, num_workers=8)

    # 2. Initialize network.
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    dev = torch.device(dev)

    net = ImitationCNN()
    net.to(dev)

    # 3. Define loss function / optimizer.
    ship_action_weights = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0]
    ship_action_weights = torch.FloatTensor(ship_action_weights).to(dev)
    ship_action_ce = nn.CrossEntropyLoss(weight=ship_action_weights)
    shipyard_action_weights = [0.1, 1.0]
    shipyard_action_weights = torch.FloatTensor(shipyard_action_weights).to(dev)
    shipyard_action_ce = nn.CrossEntropyLoss(weight=shipyard_action_weights)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 4. Train the network
    tensorboard_base_dir = './tensorboard_logs/'
    timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    tensorboard_dir = os.path.join(tensorboard_base_dir, f"model-name_{timestamp}")
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    idx = 0
    val_freq = 1
    for epoch in range(10):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            state, ship_actions, shipyard_actions = batch

            # zero the parameter gradients.
            optimizer.zero_grad()

            # forward + backward + optimize.
            state = state.to(dev)
            start = time.time()
            outputs = net(state)
            ship_action_loss = ship_action_ce(outputs[:, :6, :, :], ship_actions.to(dev))
            shipyard_action_loss = shipyard_action_ce(outputs[:, 6:, :, :], shipyard_actions.to(dev))
            loss = ship_action_loss + shipyard_action_loss
            loss.backward()
            optimizer.step()

            # print statistics.
            running_loss += loss.item()
            stats_freq = 10
            if i % stats_freq == stats_freq-1:
                # Don't log first set of batches to tensorboard, as the visualizations
                # don't scale well when the first point is an outlier.
                if i > stats_freq:
                    tensorboard_writer.add_scalar('Loss/train', running_loss / stats_freq, idx)
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / stats_freq:.5f}")
                running_loss = 0.0

            # Count of training steps
            idx += 1

        # Run validation.
        if epoch % val_freq == val_freq - 1:
            print("Running validation...")
            running_ship_cm = np.zeros((6, 6)) # running_ship_cm[gt, pred]
            running_shipyard_cm = np.zeros((2, 2)) # running_shipyard_cm[gt, pred]
            validation_loss = 0.0
            for i, batch in enumerate(val_loader):
                state, ship_actions, shipyard_actions = batch

                outputs = net(state.to(dev))
                ship_action_loss = ship_action_ce(outputs[:, :6, :, :], ship_actions.to(dev))
                shipyard_action_loss = shipyard_action_ce(outputs[:, 6:, :, :], shipyard_actions.to(dev))
                loss = ship_action_loss + shipyard_action_loss

                validation_loss += loss.item()
                outputs = outputs.detach().cpu().numpy()
                start = time.time()
                update_running_confusion_matrix(outputs[:, :6, :, :], ship_actions, running_ship_cm)
                update_running_confusion_matrix(outputs[:, 6:, :, :], shipyard_actions, running_shipyard_cm)
                print(f"time for confusion : {time.time() - start}")
                if i % stats_freq == 0:
                    print(f"Validation batch {i}...")

            tensorboard_writer.add_scalar('Loss/val', validation_loss / (i+1), idx)
            for action_idx in range(running_ship_cm.shape[0]):
                tensorboard_writer.add_scalar(
                    f'per_class_precision_SHIP_{SHIP_ACTION_ID_TO_NAME[action_idx]}/val',
                    running_ship_cm[action_idx, action_idx] / np.sum(running_ship_cm[:, action_idx]),
                    idx,
                )
                tensorboard_writer.add_scalar(
                    f'per_class_recall_SHIP_{SHIP_ACTION_ID_TO_NAME[action_idx]}/val',
                    running_ship_cm[action_idx, action_idx] / np.sum(running_ship_cm[action_idx, :]),
                    idx,
                )

            for action_idx in range(running_shipyard_cm.shape[0]):
                tensorboard_writer.add_scalar(
                    f'per_class_precision_SHIPYARD_{SHIPYARD_ACTION_ID_TO_NAME[action_idx]}/val',
                    running_shipyard_cm[action_idx, action_idx] / np.sum(running_shipyard_cm[:, action_idx]),
                    idx,
                )
                tensorboard_writer.add_scalar(
                    f'per_class_recall_SHIPYARD_{SHIPYARD_ACTION_ID_TO_NAME[action_idx]}/val',
                    running_shipyard_cm[action_idx, action_idx] / np.sum(running_shipyard_cm[action_idx, :]),
                    idx,
                )
            print(f"[{epoch + 1}] validation loss: {validation_loss / (i+1)}")
