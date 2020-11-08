import json
import math
import random

import torch

from halite_rl.utils import HaliteStateActionPair


class HaliteStateActionDataset(torch.utils.data.IterableDataset):
    def __init__(self, episode_files, team_name):
        super().__init__()
        self.episode_files = episode_files
        self._team_name = team_name

    def _load_examples_from_episode(self, episode_file):
        with open(episode_file) as f:
            episode_json = json.load(f)

        replay = json.loads(episode_json["replay"])

        # Determine whether team of interest is player 0 or 1.
        # Raises exception if neither team name matches.
        team_idx = replay["info"]["TeamNames"].index(self._team_name)

        for step_idx, step in enumerate(replay['steps']):
            hsap = HaliteStateActionPair(
                observation=replay['steps'][step_idx-1][0]['observation'],
                configuration=replay['configuration'],
                next_actions=[step[0]["action"], step[1]["action"]],
                cur_team_idx=team_idx,
            )
            state = hsap.to_state_array()
            ship_actions, shipyard_actions = hsap.to_action_arrays()
            yield (state, ship_actions, shipyard_actions)

    def shuffle(self):
        random.shuffle(self.episode_files)

    def __iter__(self):
        for episode_file in self.episode_files:
            yield from self._load_examples_from_episode(episode_file)


def hsap_worker_init_fn(worker_id):
    """worker_init_fn (see torch docs) that assigns a subset of the episode_files to each worker.

    Note: Since the number of examples per episode is not consistent, and the number of episode files
    may not be evenly divisible, it is unlikely that this will result in an even split across the
    workers. This split leads to some inefficiency at the end of the epoch when a single worker is
    forced to process the last few examples alone.
    """
    worker_info = torch.utils.data.get_worker_info()
    worker_dataset = worker_info.dataset  # the dataset copy in this worker process
    worker_id = worker_info.id
    num_workers = worker_info.num_workers

    # Assign a subset of the episode files to each worker.
    episode_files = worker_dataset.episode_files
    per_worker = int(math.ceil(len(episode_files) / num_workers))
    start_idx = per_worker * worker_id
    worker_dataset.episode_files = episode_files[start_idx:start_idx+per_worker]
