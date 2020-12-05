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
    HaliteStateActionPair,
    point_to_ji,
    SHIP_ACTION_ID_TO_ACTION,
    SHIPYARD_ACTION_ID_TO_ACTION,
)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Agent:
    """An agent wrapper around a trained model for interacting with the Halite environment.
    Actions are sampled based on probabilities predicted by the model within the constraints
    of the game rules.
    """

    def __init__(self, config, sample_actions=True):
        checkpoint = torch.load(config["CHECKPOINT_PATH"])
        self._model = HaliteActorCriticCNN(
            num_actions=config["NUM_SHIP_ACTIONS"] + config["NUM_SHIPYARD_ACTIONS"],
            input_hw=config["BOARD_HW"],
        )
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._sample_actions = sample_actions

        self._dev = self._select_device()
        self._model.to(self._dev)
        self._model.eval()
        self._config = config

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
        hsap = HaliteStateActionPair(board=board, cur_team_id=board.current_player.id)
        state = hsap.to_state_array()
        state_batch = np.expand_dims(state, axis=0)
        state_batch = torch.from_numpy(state_batch)
        state_batch = state_batch.to(self._dev)

        with torch.no_grad():
            actions, _ = self._model(state_batch)
        actions = actions.detach().cpu().numpy()

        # ship_actions/shipyard_actions will have dimensions CHW.
        ship_actions = actions[0, :self._config["NUM_SHIP_ACTIONS"], :, :]
        shipyard_actions = actions[0, self._config["NUM_SHIP_ACTIONS"]:, :, :]

        size = board.configuration["size"]
        for ship in board.current_player.ships:
            j, i = point_to_ji(ship.position, size)
            if self._sample_actions:
                ship_action_id = np.random.choice(ship_actions.shape[0], p=softmax(ship_actions[:, j, i]))
            else:
                ship_action_id = np.argmax(ship_actions[:, j, i])
            ship.next_action = SHIP_ACTION_ID_TO_ACTION.get(ship_action_id, None)

        for shipyard in board.current_player.shipyards:
            j, i = point_to_ji(shipyard.position, size)
            if self._sample_actions:
                shipyard_action_id = np.random.choice(shipyard_actions.shape[0], p=softmax(shipyard_actions[:, j, i]))
            else:
                shipyard_action_id = np.argmax(shipyard_actions[:, j, i])
            shipyard.next_action = SHIPYARD_ACTION_ID_TO_ACTION.get(shipyard_action_id, None)
