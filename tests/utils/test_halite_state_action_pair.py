import copy
import os
import tempfile

import numpy as np
import pytest
import json

from kaggle_environments.envs.halite.helpers import Board

from halite_rl.utils import HaliteStateActionPair


@pytest.fixture
def halite_sap():
    base_path = os.path.dirname(os.path.realpath(__file__))
    sample_file = os.path.join(base_path, "../samples/3602885.json")
    with open(sample_file) as f:
        episode_json = json.load(f)

    replay = json.loads(episode_json["replay"])

    team_name = "Stanley Zheng"
    team_idx = replay["info"]["TeamNames"].index(team_name)

    # This step was specifically chosen because it contains a Shipyard.SPAWN action.
    step_idx = 202
    hsap = HaliteStateActionPair(
        board=Board(
            raw_observation=replay['steps'][step_idx-1][0]['observation'],
            raw_configuration=replay['configuration'],
            next_actions=[replay['steps'][step_idx][0]["action"], replay['steps'][step_idx][1]["action"]],
        ),
        cur_team_id=team_idx,
    )
    return hsap

def all_zero_or_one(arr):
    return np.all(np.isclose(arr, 0.0) | np.isclose(arr, 1.0))

def test_to_state_array(halite_sap):
    """Sanity check of to_state_array result. Not rigorous by any means,
    but tests the basic functionality.
    """
    state = halite_sap.to_state_array()

    board_size = 21

    assert state.shape[-1] == 9

    # Channel 0: Halite map.
    assert np.all(state[:, :, 0] >= 0)
    assert np.all(state[:, :, 0] <= 500) # Max halite per cell

    # Channel 1: Current player ships.
    assert all_zero_or_one(state[:, :, 1])
    ship_coverage = np.count_nonzero(state[:, :, 1]) / (board_size ** 2)
    assert 0.001 < ship_coverage < 0.25 # Assert reasonable number of ships

    # Channel 2: Current player shipyard.
    assert all_zero_or_one(state[:, :, 2])
    shipyard_coverage = np.count_nonzero(state[:, :, 2]) / (board_size ** 2)
    assert 0.001 < shipyard_coverage < 0.05 # Assert reasonable number of shipyards

    # Channel 3: Opposing player ships.
    assert all_zero_or_one(state[:, :, 3])
    ship_coverage = np.count_nonzero(state[:, :, 3]) / (board_size ** 2)
    assert 0.001 < ship_coverage < 0.25 # Assert reasonable number of ships
    # Current player and opposing player have no overlapping ships.
    assert np.all((state[:, :, 1] + state[:, :, 3]) < 1.1)

    # Channel 4: Opposing player shipyard.
    assert all_zero_or_one(state[:, :, 4])
    shipyard_coverage = np.count_nonzero(state[:, :, 4]) / (board_size ** 2)
    assert 0.001 < shipyard_coverage < 0.05 # Assert reasonable number of shipyards
    # Current player and opposing player have no overlapping shipyards.
    assert np.all((state[:, :, 2] + state[:, :, 4]) < 1.1)

    # Channel 5: Onboard ship halite.
    assert np.all(state[:, :, 5] >= 0)
    ship_map = state[:, :, 1] + state[:, :, 3]
    # All locations with no ship have 0 halite.
    assert np.allclose(state[:, :, 5][ship_map < 0.5], 0.0)
    assert np.sum(state[:, :, 5]) > 10.0 # Not all 0.0.

    # Channel 6: Current player halite total.
    assert np.isclose(np.std(state[:, :, 6]), 0.0) # All values are the same.
    assert not np.isclose(state[0, 0, 6], 0.0) # Value is not 0.0.

    # Channel 7: Opposing player halite total.
    assert np.isclose(np.std(state[:, :, 7]), 0.0) # All values are the same.
    assert not np.isclose(state[0, 0, 7], 0.0) # Value is not 0.0.
    # The current halite totals of the 2 players should not be equal (unlikely coincidence).
    assert state[0, 0, 6] != state[0, 0, 7]

    # Channel 8: Remaining time steps.
    assert np.isclose(np.std(state[:, :, 8]), 0.0) # All values are the same.
    assert state[0, 0, 8] == (400 - 202 - 1)


def test_to_action_arrays(halite_sap):
    """Sanity check of to_action_array result. Not rigorous by any means,
    but tests the basic functionality.
    """
    ship_actions, shipyard_actions = halite_sap.to_action_arrays()

    board_size = 21
    num_cells = board_size ** 2

    assert list(ship_actions.shape) == [board_size, board_size]
    assert ship_actions.dtype == np.uint8
    # Most (but not all) cells should have no action.
    assert (num_cells * 0.95) < np.count_nonzero(ship_actions == 0) < num_cells


    assert list(shipyard_actions.shape) == [board_size, board_size]
    assert shipyard_actions.dtype == np.uint8
    # Most (but not all) cells should have no action.
    assert (num_cells * 0.95) < np.count_nonzero(shipyard_actions == 0) < num_cells


def test_encode_decode_roundtrip(halite_sap):
    """Test that encoding to file followed by decoding from file
    results in an equivalent HaliteStateActionPair.
    """
    with tempfile.TemporaryFile(mode='w+') as temp:
        halite_sap.to_json_file(temp)
        temp.seek(0)
        halite_sap_copy = HaliteStateActionPair.from_json_file(temp)

    assert np.array_equal(halite_sap.to_state_array(), halite_sap_copy.to_state_array())

    ship_actions_1, shipyard_actions_1 = halite_sap.to_action_arrays()
    ship_actions_2, shipyard_actions_2 = halite_sap_copy.to_action_arrays()
    assert np.array_equal(ship_actions_1, ship_actions_2)
    assert np.array_equal(shipyard_actions_1, shipyard_actions_2)

    assert halite_sap._cur_team_id == halite_sap_copy._cur_team_id
