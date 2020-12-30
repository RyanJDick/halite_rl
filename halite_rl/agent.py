import numpy as np
import torch
from kaggle_environments.envs.halite.helpers import (
    Board,
    Point,
    ShipAction,
    ShipyardAction,
)

from halite_rl.utils import (
    HaliteActorCriticCNN,
    board_to_state_array,
    update_board_with_actions,
    point_to_ji,
    SHIP_ACTION_ID_TO_ACTION,
    SHIPYARD_ACTION_ID_TO_ACTION,
)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sample_from_action_arrays(board, cur_team_id, ship_action_logits, shipyard_action_logits):
    """Sample an action at each location of interest.

    Arguments:
    ----------
    board : Board
        Halite board that actions will be applied to. (Used to save time and only sample actions at
        locations with either a ship or shipyard.)
    cur_team_id : str
        id of team to sample actions for.
    ship_action_logits : np.ndarray (CxHxW)
        Ship action logits from prediction model. Note CHW dimensioning.
    shipyard_action_logits : np.ndarray (CxHxW)
        Shipyard action logits from prediction model. Note CHW dimensioning.

    Returns:
    --------
    (ship_action_ids, shipyard_action_ids)
        ship_action_ids : np.ndarray
            2D array of integer ship action idxs.
        shipyard_action_ids : np.ndarray
            2D array of integer shipyard actions idxs.
    """
    size = board.configuration["size"]
    ship_actions = np.zeros((size, size), dtype=int)
    shipyard_actions = np.zeros((size, size), dtype=int)
    for ship in board.players[cur_team_id].ships:
        j, i = point_to_ji(ship.position, size)
        ship_action_id = np.random.choice(ship_action_logits.shape[0], p=softmax(ship_action_logits[:, j, i]))
        ship_actions[j, i] = ship_action_id

    for shipyard in board.players[cur_team_id].shipyards:
        j, i = point_to_ji(shipyard.position, size)
        shipyard_action_id = np.random.choice(shipyard_action_logits.shape[0], p=softmax(shipyard_action_logits[:, j, i]))
        shipyard_actions[j, i] = shipyard_action_id

    return ship_actions, shipyard_actions


class Agent:
    """An agent wrapper around a trained model for interacting with the Halite environment.
    Actions are sampled based on probabilities predicted by the model within the constraints
    of the game rules.
    """

    def __init__(self, config, sample_actions=True):
        self._model = HaliteActorCriticCNN(
            input_hw=config["BOARD_HW"],
            num_ship_actions=config["NUM_SHIP_ACTIONS"],
            num_shipyard_actions=config["NUM_SHIPYARD_ACTIONS"],
        )
        ckpt_path = config["CHECKPOINT_PATH"]
        if ckpt_path:
            checkpoint = torch.load(ckpt_path)
            model_ckpt = checkpoint['model_state_dict']
            self._model.load_state_dict(model_ckpt)

        self._sample_actions = sample_actions

        self._dev = self._select_device()
        self._model.to(self._dev)
        self._model.eval()

    def _select_device(self):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        return torch.device(dev)

    def __call__(self, observation, configuration):
        board = Board(observation, configuration)
        self.set_board_actions(board)
        return board.current_player.next_actions

    def set_board_actions(self, board):
        """Set next_actions for board.current_player in-place.
        """
        # Convert board to state representation.
        state = board_to_state_array(board, board.current_player.id)
        state_batch = np.expand_dims(state, axis=0)
        state_batch = torch.from_numpy(state_batch)
        state_batch = state_batch.to(self._dev)

        with torch.no_grad():
            ship_action_logits, shipyard_action_logits, _ = self._model(state_batch)
        ship_action_logits = ship_action_logits.detach().cpu().numpy()
        shipyard_action_logits = shipyard_action_logits.detach().cpu().numpy()

        if self._sample_actions:
            ship_actions, shipyard_actions = sample_from_action_arrays(
                board, board.current_player.id, ship_action_logits[0, ...], shipyard_action_logits[0, ...])
        else:
            ship_actions = np.argmax(ship_action_logits[0, ...], axis=0)
            shipyard_actions = np.argmax(shipyard_action_logits[0, ...], axis=0)

        update_board_with_actions(board, board.current_player.id, ship_actions, shipyard_actions)
