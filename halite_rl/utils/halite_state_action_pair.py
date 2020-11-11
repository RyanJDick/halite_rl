import json

import numpy as np

from kaggle_environments.envs.halite.helpers import (
    Board,
    Point,
    ShipAction,
    ShipyardAction,
)

# TODO: combine with reverse mapping below
SHIP_ACTION_ID_TO_NAME = {
    0: "NONE",
    1: "NORTH",
    2: "EAST",
    3: "SOUTH",
    4: "WEST",
    5: "CONVERT",
}

SHIPYARD_ACTION_ID_TO_NAME = {
    0: "NONE",
    1: "SPAWN",
}

def point_to_ji(point : Point, board_size : int):
    """j indexes from 0 to h-1 starting in top left.
    i indexes from 0 to w-1 starting in top left.
    """
    j = board_size - point.y - 1
    i = point.x
    return j, i

class HaliteStateActionPair:
    def __init__(self, observation, configuration, next_actions, cur_team_idx):
        self._observation = observation
        self._configuration = configuration
        self._next_actions = next_actions
        self._cur_team_idx = cur_team_idx

    def to_state_array(self):
        """Return array representation of state.

        Returns:
        --------
        state : np.ndarray
            state array with following channels:
                0: halite (float)
                1: current player ship (0.0 or 1.0)
                2: current player shipyard (0.0 or 1.0)
                3: opposing player ship (0.0 or 1.0)
                4: opposing player shipyard (0.0 or 1.0)
                5: onboard ship halite (float)
                ------
                6: current player halite total (float, same value at all locations)
                7: opposing player halite total (float, same value at all locations)
                8: remaining time steps (float)
        """
        board = Board(self._observation, self._configuration, self._next_actions)
        size = board.configuration["size"]
        state = np.zeros((size, size, 9), dtype=np.float32)

        opp_team_idx = (self._cur_team_idx + 1) % 2 # 0 <-> 1

        for cell in board.cells.values():
            j, i = point_to_ji(cell.position, size)
            state[j, i, 0] = cell.halite
            if cell.ship is not None:
                if cell.ship.player_id == self._cur_team_idx:
                    state[j, i, 1] = 1.0
                elif cell.ship.player_id == opp_team_idx:
                    state[j, i, 3] = 1.0
                else:
                    raise ValueError("Unexpected player_id")
                state[j, i, 5] = cell.ship.halite
            if cell.shipyard is not None:
                if cell.shipyard.player_id == self._cur_team_idx:
                    state[j, i, 2] = 1.0
                elif cell.shipyard.player_id == opp_team_idx:
                    state[j, i, 4] = 1.0
                else:
                    raise ValueError("Unexpected player_id")

        cur_player = board.players[self._cur_team_idx]
        state[:, :, 6] = cur_player.halite

        opp_player = board.players[opp_team_idx]
        state[:, :, 7] = opp_player.halite

        num_steps = board.configuration['episodeSteps']
        # -2 because steps start at zero-index,
        # and we want remaining actions not remaining observations
        remaining_steps = num_steps - board.step - 2
        state[:, :, 8] = remaining_steps

        return state

    def to_action_arrays(self):
        """Return array representation of actions taken by self._cur_team_idx.

        Returns:
        --------
        ship_actions : np.ndarray
            int ndarray indicating the ship action taken at each cell according
            to the following mapping:
                0: NONE (mine if ship is present, or no ship present)
                1: ShipAction.NORTH
                2: ShipAction.EAST
                3: ShipAction.SOUTH
                4: ShipAction.WEST
                5: ShipAction.CONVERT

        shipyard_actions : np.ndarray
            int ndarray indicating the shipyard action taken at each cell
            according to the following mapping:
                0: NONE (do not spawn if shipyard present, or no shipyard present)
                1: ShipYardAction.SPAWN
        """
        action_to_action_id = {
            ShipAction.NORTH: 1,
            ShipAction.EAST: 2,
            ShipAction.SOUTH: 3,
            ShipAction.WEST: 4,
            ShipAction.CONVERT: 5,
            ShipyardAction.SPAWN: 1,
        }

        board = Board(self._observation, self._configuration, self._next_actions)
        size = board.configuration["size"]

        cur_player = board.players[self._cur_team_idx]

        ship_actions = np.zeros((size, size), dtype=int)
        for ship in cur_player.ships:
            if ship.next_action is not None:
                action_id = action_to_action_id[ship.next_action]
                j, i = point_to_ji(ship.position, size)
                ship_actions[j, i] = action_id

        shipyard_actions = np.zeros((size, size), dtype=int)
        for shipyard in cur_player.shipyards:
            if shipyard.next_action is not None:
                action_id = action_to_action_id[shipyard.next_action]
                j, i = point_to_ji(shipyard.position, size)
                shipyard_actions[j, i] = action_id

        return ship_actions, shipyard_actions

    def to_json_file(self, file):
        state_action_dict = {
            "observation": self._observation,
            "configuration":self._configuration,
            "next_actions": self._next_actions,
            "cur_team_idx": self._cur_team_idx,
        }

        json.dump(state_action_dict, file)

    @classmethod
    def from_json_file(cls, file):
        json_dict = json.load(file)
        return cls(
            json_dict["observation"],
            json_dict["configuration"],
            json_dict["next_actions"],
            json_dict["cur_team_idx"],
        )
