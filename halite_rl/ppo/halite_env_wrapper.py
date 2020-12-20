import numpy as np

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Board

from halite_rl.utils import (
    board_to_state_array,
    update_board_with_actions,
    SHIP_ACTION_ID_TO_ACTION,
    SHIPYARD_ACTION_ID_TO_ACTION,
)
from halite_rl.utils.rewards import calc_episode_reward

class HaliteEnvWrapper():
    """Wrapper around the Halite environment that allows multiple players to interact with the
    environment in each step, and uses state and action representations that are more convenient for
    learning. Also defines a custom reward function.
    """
    def __init__(self, configuration=None, num_players=2):
        self._configuration = configuration
        if configuration == None:
            self._configuration = {
                "episodeSteps": 400,
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
        self._player_ids = [i for i in range(self._num_players)]
        self._board = None
        self._last_return = None

    def is_in_progress(self):
        # If env has never been reset.
        if self._last_return is None:
            return False

        # If last operation was a reset, so only an observation was returned.
        if self._last_return[2] is None:
            return True

        for done in self._last_return[2].values():
            # As long as at least one agent is not done, episode is still considered "in progress".
            if not done:
                return True

        return False

    def _initialize_new_board(self):
        env = make("halite", self._configuration)
        obs = env.reset(self._num_players)
        return Board(raw_observation=obs[0]["observation"], raw_configuration=self._configuration)

    def _get_state_from_board(self):
        pid_to_state = {}
        for p_id in self._player_ids:
            pid_to_state[p_id] = board_to_state_array(self._board, p_id)
        return pid_to_state

    def _get_reward_from_board(self):
        # Current reward/termination calculations currently assume 2 players in several places, even though this is not
        # guaranteed.
        assert len(self._player_ids) == 2

        pid_to_reward = {}
        pid_to_done = {}

        # Check if we have hit the step limit.
        # TODO: confirm that this is the correct interpretation of episodeSteps. (i.e. not off by 1)
        if self._board.step >= self._board.configuration.episode_steps - 1:
            pid_to_done = {p_id: True for p_id in self._player_ids}

            # Calculate rewards based on current scores.
            for p_id in self._player_ids:
                other_p_id = [i for i in self._player_ids if i != p_id][0]

                pid_to_reward[p_id] = calc_episode_reward(
                    self._board.players[p_id].halite,
                    self._board.players[other_p_id].halite,
                )

            return pid_to_reward, pid_to_done

        # Check if player has insufficient potential to continue.
        for p_id in self._player_ids:
            halite = self._board.players[p_id].halite
            num_shipyards = len(self._board.players[p_id].shipyards)
            num_ships = len(self._board.players[p_id].ships)

            if num_ships == 0 and (num_shipyards == 0 or halite < self._board.configuration.spawn_cost):
                # Game is over, because a player has been eliminated, so only one player remaining.
                pid_to_done = {p_id: True for p_id in self._player_ids}
                # Hardcode rewards for eliminating other player.
                pid_to_reward = {p_id: -10000} # eliminated player
                other_p_id = [i for i in self._player_ids if i != p_id][0]
                pid_to_reward[other_p_id] = 10000 # winning player
                return pid_to_reward, pid_to_done

        # If we got here, then the episode is still in-progress.
        # No rewards are returned until the end of the episode.
        pid_to_done = {p_id: False for p_id in self._player_ids}
        pid_to_reward = {p_id: 0 for p_id in self._player_ids}
        return pid_to_reward, pid_to_done

    def _get_action_counts(self, player_id):
        ship_action_to_id = {v: k for k, v in SHIP_ACTION_ID_TO_ACTION.items()}
        ship_action_to_id[None] = 1 # Default action id is 1 (Mine).
        ship_action_counts = np.zeros(len(ship_action_to_id) + 1) # +1 for NO_SHIP action (currently unused).
        for ship in self._board.players[player_id].ships:
            act_i = ship_action_to_id[ship.next_action]
            ship_action_counts[act_i] += 1

        shipyard_action_to_id = {v: k for k, v in SHIPYARD_ACTION_ID_TO_ACTION.items()}
        shipyard_action_to_id[None] = 1 # Default action id is 1 (NO_ACTION)
        shipyard_action_counts = np.zeros(len(shipyard_action_to_id) + 1) # +1 for NO_SHIPYARD action (currently unused).
        for shipyard in self._board.players[player_id].shipyards:
            act_i = shipyard_action_to_id[shipyard.next_action]
            ship_action_counts[act_i] += 1

        return ship_action_counts, shipyard_action_counts

    def reset(self):
        self._board = self._initialize_new_board()
        obs = self._get_state_from_board()
        self._last_return = obs, None, None, None
        return obs

    def step(self, actions):
        step_info = {}
        for p_id in self._player_ids:
            ship_action_array, shipyard_action_array = actions[p_id]
            update_board_with_actions(self._board, p_id, ship_action_array, shipyard_action_array)
            ship_action_counts, shipyard_action_counts = self._get_action_counts(p_id)
            step_info[p_id] = {
                "ship_action_counts": ship_action_counts,
                "shipyard_action_counts": shipyard_action_counts,
            }
        self._board = self._board.next()
        reward, done = self._get_reward_from_board()

        self._last_return = self._get_state_from_board(), reward, done, step_info
        return self._last_return
