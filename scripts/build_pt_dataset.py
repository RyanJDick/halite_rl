import argparse
import os
import json

import numpy as np

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import (
    Board,
    Point,
    ShipAction,
    ShipyardAction,
)

from halite_rl.utils import HaliteStateActionPair


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-id", help="submission_id for which to download episode")
    parser.add_argument("--team-name", help="TeamName associated with submission-id")
    args = parser.parse_args()

    # Find list of episode files.
    episodes_dir = f"./data/submission_{args.submission_id}_episodes/"
    episode_files = sorted(os.listdir(episodes_dir))

    output_dir = f"./data/submission_{args.submission_id}_dataset/"
    print(f"Dataset output dir: '{output_dir}'")
    os.makedirs(output_dir, exist_ok=True)
    details_file = f"./data/submission_{args.submission_id}_dataset.json"
    example_details = []
    example_idx = 0

    # Split each episode into state-action examples.
    for ep_idx, episode_file in enumerate(episode_files):
        if ep_idx % 10 == 0:
            print(f"Processing episode {ep_idx} / {len(episode_files)}")

        episode_path = os.path.join(episodes_dir, episode_file)

        with open(episode_path) as f:
            episode_json = json.load(f)

        replay = json.loads(episode_json["replay"])

        # Determine whether team of interest is player 0 or 1.
        # Raises exception if neither team name matches.
        team_idx = replay["info"]["TeamNames"].index(args.team_name)

        for step_idx, step in enumerate(replay['steps']):
            if step_idx == 0:
                continue

            hsap = HaliteStateActionPair(
                observation=replay['steps'][step_idx-1][0]['observation'],
                configuration=replay['configuration'],
                next_actions=[step[0]["action"], step[1]["action"]],
                cur_team_idx=team_idx,
            )

            # Write state-action pair to file.
            example_file = os.path.join(output_dir, f"{example_idx:08}.json")
            with open(example_file, 'w') as f:
                hsap.to_json_file(f)
            example_idx += 1

            # Store example details record.
            example_details.append(
                {
                    'episode_file': episode_files,
                    'step_idx': step_idx,
                    'example_file': example_file,
                }
            )

    # Save dataset metadata.
    with open(details_file, 'w'):
        json.dump(example_details)
    print(f"Wrote dataset details to '{details_file}'")
