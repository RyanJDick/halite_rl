import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class HaliteActorCriticCNN(nn.Module):
    """CNN that takes as input a Halite state representation and produces:
    1) a value prediction of expected future rewards under the current policy (critic), and
    2) a probability distribution over discrete actions at every board position (actor).

    The actor and critic share the base layers of the model.
    """
    # Ideas to try:
    #   - CNN layers to include board layout in value prediction.
    #   - Boolean feature indicating whether it is the last time step (as this may be when many of the "CONVERT" actions are happening).
    #   - More manual feature extraction. ex: flag indicating if you have enough cash to spawn a new ship.
    #   - Use summary states like total number of shipyard, total number of ships, etc. for actor head.
    #   - try adding batch norm layers
    #   - add pooling
    #   - skip connection with raw inputs right before applying final 1d conv operation (particularly if applying pooling)
    #   - Another hidden layer before value prediction.
    def __init__(self, input_hw, num_ship_actions, num_shipyard_actions):
        super().__init__()

        self._input_hw = input_hw
        self._num_ship_actions = num_ship_actions
        self._num_shipyard_actions = num_shipyard_actions

        # Action-probability prediction branch:
        # Note: padding is set to maintain HxW dimensions throughout.
        self._action_conv1 = nn.Conv2d(9, 16, 5, padding=2, padding_mode="circular")
        self._action_conv2 = nn.Conv2d(16, 16, 7, padding=3, padding_mode="circular")
        self._action_conv3 = nn.Conv2d(16, 32, 7, padding=3, padding_mode="circular")
        self._action_conv4 = nn.Conv2d(32, num_ship_actions + num_shipyard_actions, 1)

        # Value-prediction layers:
        # 2 for the current score_diff and steps_remaining.
        # 10 for the values from get_summary_scalars(...).
        self._state_val_linear = nn.Linear(2 + 10, 1) 

    def forward(self, input_x):
        # TODO: should we convert NCHW back to NHWC at the end?
        x = input_x.permute(0, 3, 1, 2) # from NHWC to NCHW

        x_normalized = x.clone()
        # TODO: revisit these normalization values. May want to use a logarithmic scale for halite?
        # TODO: set normalization based on game config.
        x_normalized[:, 0, :, :] /= 500.0   # On-board halite
        x_normalized[:, 5, :, :] /= 500.0   # On-ship halite
        x_normalized[:, 6, :, :] /= 5000.0  # Current player halite total
        x_normalized[:, 7, :, :] /= 5000.0  # Opposing player halite total
        x_normalized[:, 8, :, :] /= 400.0   # Remaining time steps

        # Prepare scalars to be concatenated before final layer.
        summary_scalars = self.get_summary_scalars(x)

        summary_scalars_map = torch.unsqueeze(summary_scalars, -1)
        summary_scalars_map = torch.unsqueeze(summary_scalars_map, -1) # NxCx1x1
        h, w = self._input_hw
        summary_scalars_map = summary_scalars_map.repeat(1, 1, h, w)

        # Actor:
        x_action = x_normalized
        x_action = F.relu(self._action_conv1(x_action))
        x_action = F.relu(self._action_conv2(x_action))
        x_action = F.relu(self._action_conv3(x_action))
        #x_action = torch.cat((x_action, summary_scalars_map), 1) # Cat along channel dimension.
        action_logits = self._action_conv4(x_action)
        ship_act_logits = action_logits[:, :self._num_ship_actions, :, :]
        shipyard_act_logits = action_logits[:, self._num_ship_actions:, :, :]

        # Critic:
        score_diff_list = input_x[:, 0, 0, 6:7] - input_x[:, 0, 0, 7:8] # Shape: N, 1
        score_diff = score_diff_list[:, 0]
        steps_remaining_list = input_x[:, 0, 0, 8:9]

        x_value = torch.cat((score_diff_list, steps_remaining_list, summary_scalars), dim=1)
        value_residuals = self._state_val_linear(x_value)[:, 0] # Resulting dimension of N.
        value_preds = score_diff + value_residuals

        return ship_act_logits, shipyard_act_logits, value_preds

    def get_summary_scalars(self, x):
        """Extracts key summary pieces of information useful in making both action and value predictions.
        """
        # x should have shape NCHW.
        # Assert as reminder to updatet this if the state representation changes.
        assert x.shape[1] == 9 

        # entity_counts contains summary counts for any features that make sense to count:
        # entity_counts[0]: Total Halite on the board.
        # entity_counts[1]: Total current player ships.
        # entity_counts[2]: Total current player shipyards.
        # entity_counts[3]: Total opposing player ships.
        # entity_counts[4]: Total opposing player shipyards.
        entity_counts = torch.sum(x[:, 0:5, :, :], dim=(2, 3)) # Result shape: (N, 5)

        # Use ship locations to mask on-ship halite map for each player:
        per_player_ship_halite_map = x[:, (1, 3), :, :] * x[:, 5:6, :, :] # Result shape: (N, 2, H, W)
        # total_ship_halite[0]: Total on-ship halite for current player.
        # total_ship_halite[1]: Total on-ship halite for opposing player.
        total_ship_halite = torch.sum(per_player_ship_halite_map, dim=(2, 3)) # Result shape: (N, 2)

        # Take the input scalars unchanged:
        # input_scalars[0]: Current player halite total.
        # input_scalars[1]: Opposing player halite total.
        # input_scalars[2]: Remaining time steps.
        input_scalars = x[:, 6:9, 0, 0] # Result shape: (N, 3)

        return torch.cat((entity_counts, total_ship_halite, input_scalars), 1) # Result shape: (N, 10)

    def get_action_distribution(self, ship_action_logits, shipyard_action_logits, state):
        """Convert action logits to action distribution. Distributions at cells with no ship/shipyard are overridden
        to assign all probability to a default action.

        Note: current implementation always return cpu distributions.
        """
        # Assume that ship_action_logits/shipyard_action_logits have shape [B, C, H, W].
        ship_action_logits = ship_action_logits.permute(0, 2, 3, 1) # BCHW -> BHWC
        shipyard_action_logits = shipyard_action_logits.permute(0, 2, 3, 1) # BCHW -> BHWC

        ship_action_probs = torch.nn.functional.softmax(ship_action_logits, dim=-1)
        shipyard_action_probs = torch.nn.functional.softmax(shipyard_action_logits, dim=-1)

        # Mask cells with no ships/shipyards and force distributions to put all weight on one action.
        ship_cells = state[:, :, :, 1] > 0.5
        ship_cells = torch.unsqueeze(ship_cells, dim=-1)
        no_ship_default_dist = torch.zeros(ship_action_probs.shape[-1])
        no_ship_default_dist[1] = 1.0 # Assign all probability to MINE.
        no_ship_default_dist = no_ship_default_dist.to(ship_action_probs.device)
        ship_action_probs = torch.where(ship_cells, ship_action_probs, no_ship_default_dist)

        shipyard_cells = state[:, :, :, 2] > 0.5
        shipyard_cells = torch.unsqueeze(shipyard_cells, dim=-1)
        no_shipyard_default_dist = torch.zeros(shipyard_action_probs.shape[-1])
        no_shipyard_default_dist[1] = 1.0 # Assign all probability to NO_ACTION.
        no_shipyard_default_dist = no_shipyard_default_dist.to(shipyard_action_probs.device)
        shipyard_action_probs = torch.where(shipyard_cells, shipyard_action_probs, no_shipyard_default_dist)

        # Convert to distributions.
        ship_action_dist = torch.distributions.categorical.Categorical(probs=ship_action_probs)
        ship_action_dist = torch.distributions.independent.Independent(
            ship_action_dist, reinterpreted_batch_ndims=2)
        shipyard_action_dist = torch.distributions.categorical.Categorical(probs=shipyard_action_probs)
        shipyard_action_dist = torch.distributions.independent.Independent(
            shipyard_action_dist, reinterpreted_batch_ndims=2)

        return ship_action_dist, shipyard_action_dist

    def action_log_prob(self, ship_action_dist, shipyard_action_dist, ship_action, shipyard_action):
        return ship_action_dist.log_prob(ship_action) + shipyard_action_dist.log_prob(shipyard_action)
