import argparse
from collections import defaultdict
from datetime import datetime
import os
import time

import h5py
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
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
    PixelWeightedCrossEntropyLoss,
    plot_confusion_matrix,
)


def update_running_confusion_matrix(
    outputs,
    actions_gt,
    confusion_matrix,
    ignore_empty_squares,
):
    num_classes = outputs.shape[1]
    preds = np.argmax(outputs, axis=1) # outputs NCHW -> NHW
    non_empty = actions_gt != 0

    for gt_class in range(num_classes):
        for pred_class in range(num_classes):
            if ignore_empty_squares:
                confusion_matrix[gt_class, pred_class] += \
                    np.count_nonzero((actions_gt == gt_class) & (preds == pred_class) & non_empty)
            else:
                confusion_matrix[gt_class, pred_class] += \
                    np.count_nonzero((actions_gt == gt_class) & (preds == pred_class))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()

    # 0. Load configs.
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    # 1. Initialize datasets.
    train_dataset_file = os.path.join(config["BASE_DATA_DIR"], config["TRAIN_HDF5_FILE"])
    train_data_hdf5 = h5py.File(train_dataset_file, 'r')
    train_dataset = HaliteStateActionHDF5Dataset(train_data_hdf5)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config["BATCH_SIZE"], num_workers=8)

    val_dataset_file = os.path.join(config["BASE_DATA_DIR"], config["VAL_HDF5_FILE"])
    val_data_hdf5 = h5py.File(val_dataset_file, 'r')
    val_dataset = HaliteStateActionHDF5Dataset(val_data_hdf5)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], num_workers=8)

    # 2. Select device.
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    dev = torch.device(dev)

    # 3. Assemble model name.
    timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    model_name = f"{config['MODEL_NAME']}_{timestamp}"

    # 4. Load checkpoint if path provided.
    best_val_loss = 1e9
    start_epoch = 0
    train_examples = 0
    train_batches = 0
    checkpoint = None
    ckpt_path = config["CHECKPOINT_PATH"]
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        train_examples = checkpoint['train_examples']
        train_batches = checkpoint['train_batches']
        best_val_loss = checkpoint['val_loss']

    # 5. Initialize network.
    net = ImitationCNN(config["NUM_SHIP_ACTIONS"] + config["NUM_SHIPYARD_ACTIONS"])
    if checkpoint is not None:
        net.load_state_dict(checkpoint['model_state_dict'])
    net.to(dev)

    # 6. Define loss function / optimizer.
    if config["IGNORE_EMPTY_SQUARES"]:
        ship_action_weights = torch.FloatTensor([1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 4.0]).to(dev)
        shipyard_action_weights = torch.FloatTensor([1.0, 1.0, 1.0]).to(dev)
        ship_action_ce = PixelWeightedCrossEntropyLoss(weight=ship_action_weights)
        shipyard_action_ce = PixelWeightedCrossEntropyLoss(weight=shipyard_action_weights)
    else:
        ship_action_weights = torch.FloatTensor([0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]).to(dev)
        shipyard_action_weights = torch.FloatTensor([0.0, 1.0, 2.0]).to(dev)
        ship_action_ce = torch.nn.CrossEntropyLoss(weight=ship_action_weights)
        shipyard_action_ce = torch.nn.CrossEntropyLoss(weight=shipyard_action_weights)

    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 7. Initialize tensorboard writer.
    tensorboard_base_dir = './tensorboard_logs/'
    tensorboard_dir = os.path.join(tensorboard_base_dir, model_name)
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    print("Enter a description for this run...")
    run_description = input()
    tensorboard_writer.add_text("description", run_description)
    print("Starting training...")

    # 8. Train the network
    stats_freq_batches = 100
    val_freq_epochs = 5
    for epoch in range(start_epoch, 1000):
        running_loss = 0.0
        running_batch_count = 0
        for i, batch in enumerate(train_loader):
            state, ship_actions, shipyard_actions = batch
            if state.shape[0] != config["BATCH_SIZE"]:
                print(f"Incomplete batch, skipping...")
                continue
            state = state.to(dev)
            ship_actions = ship_actions.long().to(dev)
            shipyard_actions = shipyard_actions.long().to(dev)

            # zero the parameter gradients.
            optimizer.zero_grad()

            # forward + backward + optimize.
            outputs = net(state)
            if config["IGNORE_EMPTY_SQUARES"]:
                ship_action_loss = ship_action_ce(
                    outputs[:, :config["NUM_SHIP_ACTIONS"], :, :],
                    ship_actions,
                    (ship_actions != 0).type(torch.uint8),
                )
                shipyard_action_loss = shipyard_action_ce(
                    outputs[:, config["NUM_SHIP_ACTIONS"]:, :, :],
                    shipyard_actions,
                    (shipyard_actions != 0).type(torch.uint8),
                )
            else:
                ship_action_loss = ship_action_ce(
                    outputs[:, :config["NUM_SHIP_ACTIONS"], :, :], ship_actions)
                shipyard_action_loss = shipyard_action_ce(
                    outputs[:, config["NUM_SHIP_ACTIONS"]:, :, :], shipyard_actions)
            loss = ship_action_loss + shipyard_action_loss
            loss.backward()
            optimizer.step()

            # print statistics.
            running_loss += loss.item()
            running_batch_count += 1
            train_examples += state.shape[0]
            train_batches += 1

            if train_batches % stats_freq_batches == 0 and train_batches != 0:
                tensorboard_writer.add_scalar(
                    'Loss/train', running_loss / running_batch_count, train_batches)
                print(f"[{epoch + 1}, {i + 1:5d}] batches: {train_batches}, examples: {train_examples}, loss: {running_loss / running_batch_count:.8f}")
                running_loss = 0.0
                running_batch_count = 0

        # Run validation.
        if epoch % val_freq_epochs == val_freq_epochs - 1:
            print("Running validation...")
            running_ship_cm = np.zeros((config["NUM_SHIP_ACTIONS"], config["NUM_SHIP_ACTIONS"])) # running_ship_cm[gt, pred]
            running_shipyard_cm = np.zeros((config["NUM_SHIPYARD_ACTIONS"], config["NUM_SHIPYARD_ACTIONS"])) # running_shipyard_cm[gt, pred]
            val_loss = 0.0
            val_batch_count = 0
            for i, batch in enumerate(val_loader):
                state, ship_actions, shipyard_actions = batch
                if state.shape[0] != config["BATCH_SIZE"]:
                    print(f"Incomplete batch, skipping...")
                    continue
                state = state.to(dev)
                ship_actions_dev = ship_actions.long().to(dev)
                shipyard_actions_dev = shipyard_actions.long().to(dev)

                outputs = net(state)
                if config["IGNORE_EMPTY_SQUARES"]:
                    ship_action_loss = ship_action_ce(
                        outputs[:, :config["NUM_SHIP_ACTIONS"], :, :],
                        ship_actions_dev,
                        (ship_actions_dev != 0).type(torch.uint8),
                    )
                    shipyard_action_loss = shipyard_action_ce(
                        outputs[:, config["NUM_SHIP_ACTIONS"]:, :, :],
                        shipyard_actions_dev,
                        (shipyard_actions_dev != 0).type(torch.uint8),
                    )
                else:
                    ship_action_loss = ship_action_ce(
                        outputs[:, :config["NUM_SHIP_ACTIONS"], :, :], ship_actions_dev)
                    shipyard_action_loss = shipyard_action_ce(
                        outputs[:, config["NUM_SHIP_ACTIONS"]:, :, :], shipyard_actions_dev)
                loss = ship_action_loss + shipyard_action_loss

                val_loss += loss.item()
                val_batch_count += 1
                outputs = outputs.detach().cpu().numpy()
                update_running_confusion_matrix(
                    outputs[:, :config["NUM_SHIP_ACTIONS"], :, :],
                    ship_actions,
                    running_ship_cm,
                    config["IGNORE_EMPTY_SQUARES"],
                )
                update_running_confusion_matrix(
                    outputs[:, config["NUM_SHIP_ACTIONS"]:, :, :],
                    shipyard_actions,
                    running_shipyard_cm,
                    config["IGNORE_EMPTY_SQUARES"],
                )
                if i % stats_freq_batches == 0:
                    print(f"Validation batch {i}...")

            tensorboard_writer.add_scalar('Loss/val', val_loss / val_batch_count, train_batches)
            for action_idx in range(running_ship_cm.shape[0]):
                tensorboard_writer.add_scalar(
                    f'per_class_precision_SHIP_{SHIP_ACTION_ID_TO_NAME[action_idx]}/val',
                    running_ship_cm[action_idx, action_idx] / np.sum(running_ship_cm[:, action_idx]),
                    train_batches,
                )
                tensorboard_writer.add_scalar(
                    f'per_class_recall_SHIP_{SHIP_ACTION_ID_TO_NAME[action_idx]}/val',
                    running_ship_cm[action_idx, action_idx] / np.sum(running_ship_cm[action_idx, :]),
                    train_batches,
                )

            for action_idx in range(running_shipyard_cm.shape[0]):
                tensorboard_writer.add_scalar(
                    f'per_class_precision_SHIPYARD_{SHIPYARD_ACTION_ID_TO_NAME[action_idx]}/val',
                    running_shipyard_cm[action_idx, action_idx] / np.sum(running_shipyard_cm[:, action_idx]),
                    train_batches,
                )
                tensorboard_writer.add_scalar(
                    f'per_class_recall_SHIPYARD_{SHIPYARD_ACTION_ID_TO_NAME[action_idx]}/val',
                    running_shipyard_cm[action_idx, action_idx] / np.sum(running_shipyard_cm[action_idx, :]),
                    train_batches,
                )

            # Write confusion matrices to tensorboard.
            ship_cm_img = plot_confusion_matrix(
                running_ship_cm,
                [SHIP_ACTION_ID_TO_NAME[i] for i in range(config["NUM_SHIP_ACTIONS"])],
            )
            print(ship_cm_img.shape)
            tensorboard_writer.add_image(
                "confusion_matrix_SHIP/val", ship_cm_img, train_batches, dataformats="HWC")
            shipyard_cm_img = plot_confusion_matrix(
                running_shipyard_cm,
                [SHIPYARD_ACTION_ID_TO_NAME[i] for i in range(config["NUM_SHIPYARD_ACTIONS"])],
            )
            tensorboard_writer.add_image(
                "confusion_matrix_SHIPYARD/val", shipyard_cm_img, train_batches, dataformats="HWC")

            print(f"[{epoch + 1}] ({train_batches}) validation loss: {val_loss / val_batch_count}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = f"./checkpoints/{model_name}/ckpt_epoch{epoch}.pt"
                print(f"New low validation loss. Saving checkpoint to '{ckpt_path}'")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'train_examples': train_examples,
                    'train_batches': train_batches,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, ckpt_path)
