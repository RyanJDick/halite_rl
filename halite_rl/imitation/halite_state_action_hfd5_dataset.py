import torch


class HaliteStateActionHDF5Dataset(torch.utils.data.Dataset):

    def __init__(self, hdf5_file):
        self._hdf5_file = hdf5_file
        self._example_paths = []
        for episode_name, episode in self._hdf5_file.items():
            for step_id in episode.keys():
                self._example_paths.append(f"{episode_name}/{step_id}")
        self._example_paths.sort()

    def __len__(self):
        return len(self._example_paths)

    def __getitem__(self, idx):
        example = self._hdf5_file[self._example_paths[idx]]
        # ellipsis indexing converts from hdf5 dataset to np.ndarray.
        state = example["state"][...]
        ship_actions = example["ship_actions"][...]
        shipyard_actions = example["shipyard_actions"][...]
        return (state, ship_actions, shipyard_actions)
