import argparse
import glob
import json
import os
import random

import h5py
import numpy as np
import yaml

from kaggle_environments.envs.halite.helpers import Board

from halite_rl.utils import HaliteStateActionPair


def load_episode_info(episode_file, team_name):
    """Load summary episode info. (Intended to be used in calculating episode reward.)
    """
    with open(episode_file) as f:
        episode_json = json.load(f)

    replay = json.loads(episode_json["replay"])

    # Determine whether team of interest is player 0 or 1.
    # Raises exception if neither team name matches.
    team_idx = replay["info"]["TeamNames"].index(team_name)
    other_team_idx = (team_idx + 1) % 2

    # Reward is defined in replay["specification"] as:
    # "The amount of player owned halite (equal to players[index][0]) if the player
    #  has not been eliminated, else step_eliminated - episode_steps - 1."
    cur_team_reward = replay["rewards"][team_idx] or 0 # Rewards are occasionally None, not sure why.
    other_team_reward = replay["rewards"][other_team_idx] or 0
    cur_team_won = cur_team_reward > other_team_reward

    cur_team_score = replay["steps"][0][0]["observation"]["players"][team_idx][0]
    other_team_score = replay["steps"][0][0]["observation"]["players"][other_team_idx][0]

    max_steps = replay["configuration"]["episodeSteps"]
    steps = len(replay["steps"])
    ended_early = steps < max_steps
    return cur_team_won, ended_early, cur_team_score, other_team_score, cur_team_reward, other_team_reward

def load_examples_from_episode(episode_file, team_name):
    with open(episode_file) as f:
        episode_json = json.load(f)

    replay = json.loads(episode_json["replay"])

    # Determine whether team of interest is player 0 or 1.
    # Raises exception if neither team name matches.
    team_idx = replay["info"]["TeamNames"].index(team_name)

    for step_idx, step in enumerate(replay['steps']):
        hsap = HaliteStateActionPair(
            board=Board(
                raw_observation=replay['steps'][step_idx-1][0]['observation'],
                raw_configuration=replay['configuration'],
                next_actions=[step[0]["action"], step[1]["action"]],
            ),
            cur_team_id=team_idx,
        )
        state = hsap.to_state_array()
        ship_actions, shipyard_actions = hsap.to_action_arrays()
        yield (state, ship_actions, shipyard_actions)


def build_dataset(episodes, team_name, filename):
    with h5py.File(filename, "w") as f:
        for epi_idx, episode_file in enumerate(episodes):
            print(f"{epi_idx}/{len(episodes) - 1}: {episode_file}")
            example_idx = 0

            cur_team_won, ended_early, cur_team_score, other_team_score, cur_team_reward, other_team_reward = load_episode_info(episode_file, team_name)

            episode_id = os.path.splitext(os.path.basename(episode_file))[0]
            episode_group = f.create_group(episode_id)
            episode_group.attrs["cur_team_won"] = cur_team_won
            episode_group.attrs["cur_team_score"] = cur_team_score
            episode_group.attrs["other_team_score"] = other_team_score
            episode_group.attrs["ended_early"] = ended_early
            episode_group.attrs["cur_team_reward"] = cur_team_reward
            episode_group.attrs["other_team_reward"] = other_team_reward

            for example in load_examples_from_episode(episode_file, team_name):
                state, ship_actions, shipyard_actions = example

                episode_group.create_dataset(f"{example_idx}/state", data=state)
                episode_group.create_dataset(f"{example_idx}/ship_actions", data=ship_actions)
                episode_group.create_dataset(f"{example_idx}/shipyard_actions", data=shipyard_actions)

                example_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--train-frac", type=float, default=0.7)
    args = parser.parse_args()

    # 0. Load configs.
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    episode_files = glob.glob(os.path.join(config["BASE_DATA_DIR"], "*.json"))
    random.Random(123).shuffle(episode_files) # Seed for consistent behaviour.

    split_idx = int(len(episode_files) * args.train_frac)
    train_files = sorted(episode_files[:split_idx])
    val_files = sorted(episode_files[split_idx:])

    train_dataset_file = os.path.join(config["BASE_DATA_DIR"], config["TRAIN_HDF5_FILE"])
    build_dataset(train_files, config["TEAM_NAME"], train_dataset_file)
    val_dataset_file = os.path.join(config["BASE_DATA_DIR"], config["VAL_HDF5_FILE"])
    build_dataset(val_files, config["TEAM_NAME"], val_dataset_file)
