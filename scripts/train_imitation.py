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
    HaliteStateActionHDF5Dataset,
)
from halite_rl.utils import (
    HaliteActorCriticCNN,
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

def update_running_state_value_mae_list(
    state_value_pred,
    state_value,
    running_state_value_mae,
):
    abs_errs = np.abs(state_value_pred - state_value)
    running_state_value_mae.append(abs_errs.mean())

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
    net = HaliteActorCriticCNN(
        input_hw=config["BOARD_HW"],
        num_ship_actions=config["NUM_SHIP_ACTIONS"],
        num_shipyard_actions=config["NUM_SHIPYARD_ACTIONS"],
    )
    if checkpoint is not None:
        net.load_state_dict(checkpoint['model_state_dict'])
    net.to(dev)

    # 6. Define loss function / optimizer.
    if config["IGNORE_EMPTY_SQUARES"]:
        ship_action_weights = torch.FloatTensor(config["SHIP_ACTION_LOSS_WEIGHTS"]).to(dev)
        shipyard_action_weights = torch.FloatTensor(config["SHIPYARD_ACTION_LOSS_WEIGHTS"]).to(dev)
        ship_action_ce = PixelWeightedCrossEntropyLoss(weight=ship_action_weights)
        shipyard_action_ce = PixelWeightedCrossEntropyLoss(weight=shipyard_action_weights)
    else:
        ship_action_weights = torch.FloatTensor(config["SHIP_ACTION_LOSS_WEIGHTS"]).to(dev)
        shipyard_action_weights = torch.FloatTensor(config["SHIPYARD_ACTION_LOSS_WEIGHTS"]).to(dev)
        ship_action_ce = torch.nn.CrossEntropyLoss(weight=ship_action_weights)
        shipyard_action_ce = torch.nn.CrossEntropyLoss(weight=shipyard_action_weights)
    state_value_mse = torch.nn.MSELoss()

    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 7. Initialize tensorboard writer.
    tensorboard_base_dir = './tensorboard_logs_imitation/'
    tensorboard_dir = os.path.join(tensorboard_base_dir, model_name)
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    print("Enter a description for this run...")
    run_description = input()
    tensorboard_writer.add_text("description", run_description)
    print("Starting training...")

    # 8. Train the network
    stats_freq_batches = 100
    val_freq_epochs = 5
    for epoch in range(start_epoch, 10000):
        running_loss = 0.0
        running_ship_action_loss = 0.0
        running_shipyard_action_loss = 0.0
        running_state_value_loss = 0.0
        running_batch_count = 0
        for i, batch in enumerate(train_loader):
            net.train() # Switch to train mode. (Dropout enabled)
            state, ship_actions, shipyard_actions, state_value = batch
            if state.shape[0] != config["BATCH_SIZE"]:
                print(f"Incomplete batch, skipping...")
                continue
            state = state.to(dev)
            ship_actions = ship_actions.long().to(dev)
            shipyard_actions = shipyard_actions.long().to(dev)
            state_value = state_value.float().to(dev)

            # zero the parameter gradients.
            optimizer.zero_grad()

            # forward + backward + optimize.
            ship_act_logits, shipyard_act_logits, value_preds = net(state)
            if config["IGNORE_EMPTY_SQUARES"]:
                ship_action_loss = ship_action_ce(
                    ship_act_logits,
                    ship_actions,
                    (ship_actions != 0).type(torch.uint8),
                )
                shipyard_action_loss = shipyard_action_ce(
                    shipyard_act_logits,
                    shipyard_actions,
                    (shipyard_actions != 0).type(torch.uint8),
                )
            else:
                ship_action_loss = ship_action_ce(ship_act_logits, ship_actions)
                shipyard_action_loss = shipyard_action_ce(shipyard_act_logits, shipyard_actions)

            state_value_loss = state_value_mse(value_preds, state_value)

            loss = ship_action_loss + \
                config["SHIPYARD_LOSS_WEIGHT"] * shipyard_action_loss + \
                config["STATE_VALUE_LOSS_WEIGHT"] * state_value_loss
            loss.backward()
            optimizer.step()

            # print statistics.
            running_loss += loss.item()
            running_ship_action_loss += ship_action_loss.item()
            running_shipyard_action_loss += shipyard_action_loss.item()
            running_state_value_loss += state_value_loss.item()
            running_batch_count += 1
            train_examples += state.shape[0]
            train_batches += 1

            if train_batches % stats_freq_batches == 0 and train_batches != 0:
                tensorboard_writer.add_scalar('Loss/train', running_loss / running_batch_count, train_batches)
                tensorboard_writer.add_scalar('Loss_ship_action/train', running_ship_action_loss / running_batch_count, train_batches)
                tensorboard_writer.add_scalar('Loss_shipyard_action/train', running_shipyard_action_loss / running_batch_count, train_batches)
                tensorboard_writer.add_scalar('Loss_state_value/train', running_state_value_loss / running_batch_count, train_batches)
                print(f"[{epoch + 1}, {i + 1:5d}] batches: {train_batches}, examples: {train_examples}, loss: {running_loss / running_batch_count:.8f}")
                running_loss = 0.0
                running_ship_action_loss = 0.0
                running_shipyard_action_loss = 0.0
                running_state_value_loss = 0.0
                running_batch_count = 0

        # Run validation.
        if epoch % val_freq_epochs == val_freq_epochs - 1:
            print("Running validation...")
            net.eval() # Disable dropout during validation.
            running_ship_cm = np.zeros((config["NUM_SHIP_ACTIONS"], config["NUM_SHIP_ACTIONS"])) # running_ship_cm[gt, pred]
            running_shipyard_cm = np.zeros((config["NUM_SHIPYARD_ACTIONS"], config["NUM_SHIPYARD_ACTIONS"])) # running_shipyard_cm[gt, pred]
            running_state_value_mae = []
            running_val_loss = 0.0
            running_val_ship_action_loss = 0.0
            running_val_shipyard_action_loss = 0.0
            running_val_state_value_loss = 0.0
            val_batch_count = 0
            for i, batch in enumerate(val_loader):
                state, ship_actions, shipyard_actions, state_value = batch
                if state.shape[0] != config["BATCH_SIZE"]:
                    print(f"Incomplete batch, skipping...")
                    continue
                state = state.to(dev)
                ship_actions_dev = ship_actions.long().to(dev)
                shipyard_actions_dev = shipyard_actions.long().to(dev)
                state_value_dev = state_value.float().to(dev)

                # TODO: should not have so much code duplication here. Move loss calculation out into function.
                ship_act_logits, shipyard_act_logits, value_preds  = net(state)
                if config["IGNORE_EMPTY_SQUARES"]:
                    ship_action_loss = ship_action_ce(
                        ship_act_logits,
                        ship_actions_dev,
                        (ship_actions_dev != 0).type(torch.uint8),
                    )
                    shipyard_action_loss = shipyard_action_ce(
                        shipyard_act_logits,
                        shipyard_actions_dev,
                        (shipyard_actions_dev != 0).type(torch.uint8),
                    )
                else:
                    ship_action_loss = ship_action_ce(ship_act_logits, ship_actions_dev)
                    shipyard_action_loss = shipyard_action_ce(shipyard_act_logits, shipyard_actions_dev)
                state_value_loss = state_value_mse(value_preds, state_value_dev)

                loss = ship_action_loss + \
                    config["SHIPYARD_LOSS_WEIGHT"] * shipyard_action_loss + \
                    config["STATE_VALUE_LOSS_WEIGHT"] * state_value_loss

                running_val_loss += loss.item()
                running_val_ship_action_loss += ship_action_loss.item()
                running_val_shipyard_action_loss += shipyard_action_loss.item()
                running_val_state_value_loss += state_value_loss.item()
                val_batch_count += 1
                ship_act_logits = ship_act_logits.detach().cpu().numpy()
                shipyard_act_logits = shipyard_act_logits.detach().cpu().numpy()
                value_preds = value_preds.detach().cpu().numpy()
                update_running_confusion_matrix(
                    ship_act_logits,
                    ship_actions,
                    running_ship_cm,
                    config["IGNORE_EMPTY_SQUARES"],
                )
                update_running_confusion_matrix(
                    shipyard_act_logits,
                    shipyard_actions,
                    running_shipyard_cm,
                    config["IGNORE_EMPTY_SQUARES"],
                )
                update_running_state_value_mae_list(
                    value_preds,
                    state_value.detach().cpu().numpy(),
                    running_state_value_mae,
                )
                if i % stats_freq_batches == 0:
                    print(f"Validation batch {i}...")

            tensorboard_writer.add_scalar('Loss/val', running_val_loss / val_batch_count, train_batches)
            tensorboard_writer.add_scalar('Loss_ship_action/val', running_val_ship_action_loss / val_batch_count, train_batches)
            tensorboard_writer.add_scalar('Loss_shipyard_action/val', running_val_shipyard_action_loss / val_batch_count, train_batches)
            tensorboard_writer.add_scalar('Loss_state_value/val', running_val_state_value_loss / val_batch_count, train_batches)
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
            tensorboard_writer.add_scalar(
                "state_value_mae/val",
                np.mean(running_state_value_mae), # Mean of batch means (assumes that all batches are the same size - which is enforced above).
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

            print(f"[{epoch + 1}] ({train_batches}) validation loss: {running_val_loss / val_batch_count}")

            if running_val_loss < best_val_loss:
                best_val_loss = running_val_loss
                ckpt_path = f"./checkpoints/{model_name}/ckpt_epoch{epoch}.pt"
                print(f"New low validation loss. Saving checkpoint to '{ckpt_path}'")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'train_examples': train_examples,
                    'train_batches': train_batches,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': running_val_loss,
                }, ckpt_path)
