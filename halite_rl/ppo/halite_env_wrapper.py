from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Board

from halite_rl.utils import (
    board_to_state_array,
    update_board_with_actions,
)

class HaliteEnvWrapper():
    """Wrapper around the Halite environment that allows multiple players to interact with the
    environment in each step, and uses state and action representations that are more convenient for
    learning.
    """
    def __init__(self, configuration=None, num_players=2):
        self._configuration = configuration
        if configuration == None:
            self._configuration = {
                "startingHalite": 5000,
                "size": 21,
                "spawnCost": 500,
                "convertCost": 500,
                "moveCost": 0,
                "collectRate": 0.25,
                "regenRate": 0.02,
                "maxCellHalite": 500,
            }
        self._num_players = num_players
        self._player_ids = [str(i) for i in range(self._num_players)]
        self._board = None

    def _initialize_new_board(self):
        env = make("halite", self._configuration)
        obs = env.reset(self._num_players)
        return Board(raw_observation=obs[0], raw_configuration=self._configuration)

    def _get_state_from_board(self):
        pid_to_state = {}
        for p_id in self._player_ids:
            pid_to_state[p_id] = board_to_state_array(self._board, p_id)
        return pid_to_state

    def reset(self):
        self._board = self._initialize_new_board()
        return self._get_state_from_board()

    def step(self, actions):
        for p_id in self._player_ids:
            ship_action_array, shipyard_action_array = actions[p_id]
            update_board_with_actions(self._board, p_id, ship_action_array, shipyard_action_array)
        self._board.step()
        return self._get_state_from_board()
