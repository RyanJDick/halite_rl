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
    0: "NO_SHIP",
    1: "MINE",
    2: "NORTH",
    3: "EAST",
    4: "SOUTH",
    5: "WEST",
    6: "CONVERT",
}
SHIP_ACTION_ID_TO_ACTION = {
    2: ShipAction.NORTH,
    3: ShipAction.EAST,
    4: ShipAction.SOUTH,
    5: ShipAction.WEST,
    6: ShipAction.CONVERT,
}

SHIPYARD_ACTION_ID_TO_NAME = {
    0: "NO_SHIPYARD",
    1: "NO_ACTION",
    2: "SPAWN",
}
SHIPYARD_ACTION_ID_TO_ACTION = {
    2: ShipyardAction.SPAWN,
}

def point_to_ji(point : Point, board_size : int):
    """j indexes from 0 to h-1 starting in top left.
    i indexes from 0 to w-1 starting in top left.
    """
    j = board_size - point.y - 1
    i = point.x
    return j, i

class HaliteStateActionPair:
    def __init__(self, board, cur_team_id):
        self._board = board
        self._cur_team_id = cur_team_id

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
        size = self._board.configuration["size"]
        state = np.zeros((size, size, 9), dtype=np.float32)

        opp_team_id = (self._cur_team_id + 1) % 2 # 0 <-> 1

        for cell in self._board.cells.values():
            j, i = point_to_ji(cell.position, size)
            state[j, i, 0] = cell.halite
            if cell.ship is not None:
                if cell.ship.player_id == self._cur_team_id:
                    state[j, i, 1] = 1.0
                elif cell.ship.player_id == opp_team_id:
                    state[j, i, 3] = 1.0
                else:
                    raise ValueError("Unexpected player_id")
                state[j, i, 5] = cell.ship.halite
            if cell.shipyard is not None:
                if cell.shipyard.player_id == self._cur_team_id:
                    state[j, i, 2] = 1.0
                elif cell.shipyard.player_id == opp_team_id:
                    state[j, i, 4] = 1.0
                else:
                    raise ValueError("Unexpected player_id")

        cur_player = self._board.players[self._cur_team_id]
        state[:, :, 6] = cur_player.halite

        opp_player = self._board.players[opp_team_id]
        state[:, :, 7] = opp_player.halite

        num_steps = self._board.configuration['episodeSteps']
        # -2 because steps start at zero-index,
        # and we want remaining actions not remaining observations
        remaining_steps = num_steps - self._board.step - 2
        state[:, :, 8] = remaining_steps

        return state

    def to_action_arrays(self):
        """Return array representation of actions taken by self._cur_team_id.

        Returns:
        --------
        ship_actions : np.ndarray
            int ndarray indicating the ship action taken at each cell according
            to the following mapping:
                0: No ship present.
                1: Mine
                2: ShipAction.NORTH
                3: ShipAction.EAST
                4: ShipAction.SOUTH
                5: ShipAction.WEST
                6: ShipAction.CONVERT

        shipyard_actions : np.ndarray
            int ndarray indicating the shipyard action taken at each cell
            according to the following mapping:
                0: No shipyard present.
                1: Do not spawn.
                2: ShipYardAction.SPAWN
        """
        action_to_action_id = {
            ShipAction.NORTH: 2,
            ShipAction.EAST: 3,
            ShipAction.SOUTH: 4,
            ShipAction.WEST: 5,
            ShipAction.CONVERT: 6,
            ShipyardAction.SPAWN: 2,
        }

        size = self._board.configuration["size"]

        cur_player = self._board.players[self._cur_team_id]

        ship_actions = np.zeros((size, size), dtype=np.uint8)
        for ship in cur_player.ships:
            action_id = 1 # Assume no action, i.e. "mine".
            if ship.next_action is not None:
                action_id = action_to_action_id[ship.next_action]
            j, i = point_to_ji(ship.position, size)
            ship_actions[j, i] = action_id

        shipyard_actions = np.zeros((size, size), dtype=np.uint8)
        for shipyard in cur_player.shipyards:
            action_id = 1 # Assume no action.
            if shipyard.next_action is not None:
                action_id = action_to_action_id[shipyard.next_action]
            j, i = point_to_ji(shipyard.position, size)
            shipyard_actions[j, i] = action_id

        return ship_actions, shipyard_actions

    def to_json_file(self, file):
        # TODO: rename cur_team_idx -> cur_team_id here and in from_json_file.
        state_action_dict = {
            "observation": self._board.observation,
            "configuration": {k: self._board.configuration[k] for k in self._board.configuration.keys()}, # configuration is a ReadOnlyDict, we unwrap it into a dict.
            "next_actions": [self._board.players[i].next_actions for i in range(len(self._board.players))], # self._board.players is a dict[int: player]
            "cur_team_idx": self._cur_team_id,
        }

        json.dump(state_action_dict, file)

    @classmethod
    def from_json_file(cls, file):
        json_dict = json.load(file)
        return cls(
            Board(json_dict["observation"], json_dict["configuration"], json_dict["next_actions"]),
            json_dict["cur_team_idx"],
        )
