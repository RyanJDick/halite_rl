import numpy as np
import torch
from kaggle_environments.envs.halite.helpers import (
    Board,
    Point,
    ShipAction,
    ShipyardAction,
)

from halite_rl.imitation import ImitationCNN
from halite_rl.utils import (
    HaliteStateActionPair,
    point_to_ji,
    SHIP_ACTION_ID_TO_ACTION,
    SHIPYARD_ACTION_ID_TO_ACTION,
)


class Agent:
    """An agent wrapper around a trained model for interacting with the Halite environment.
    Actions are sampled based on probabilities predicted by the model within the constraints
    of the game rules.
    """

    def __init__(self, config):
        checkpoint = torch.load(config["CHECKPOINT_PATH"])
        self._model = ImitationCNN(config["NUM_SHIP_ACTIONS"] + config["NUM_SHIPYARD_ACTIONS"])
        self._model.load_state_dict(checkpoint['model_state_dict'])

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
            actions = self._model(state_batch)
        actions = actions.detach().cpu().numpy()

        # ship_actions/shipyard_actions will have dimensions CHW.
        ship_actions = actions[0, :self._config["NUM_SHIP_ACTIONS"], :, :]
        shipyard_actions = actions[0, self._config["NUM_SHIP_ACTIONS"]:, :, :]

        ship_actions = np.argmax(ship_actions, axis=0) # CHW -> HW
        shipyard_actions = np.argmax(shipyard_actions, axis=0) # CHW -> HW

        size = board.configuration["size"]
        for ship in board.current_player.ships:
            j, i = point_to_ji(ship.position, size)
            ship.next_action = SHIP_ACTION_ID_TO_ACTION.get(ship_actions[j, i], None)

        for shipyard in board.current_player.shipyards:
            j, i = point_to_ji(shipyard.position, size)
            shipyard.next_action = SHIPYARD_ACTION_ID_TO_ACTION.get(shipyard_actions[j, i], None)
