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
    #   - a few conv with large kernel: 21x21, followed by 1d conv for final prediction
    #   - try adding batch norm layers
    #   - add pooling
    #   - skip connection with raw inputs right before applying final 1d conv operation (particularly if applying pooling)
    #   - Another hidden layer before value prediction.
    #   - Reduce dimenionality further before linear layer with another conv, or pooling.
    #   - ***** both value-prediction and action prediction could probably benefit from feeding some summary stats into the final layer:
    #       - number of home ships, away ships, home shipyards, away shipyards
    #       - total amount of halite mined, total amount of halite on ships
    #       - number of steps remaining
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
        # 10 channels will be added from get_summary_scalars(...).
        self._action_conv4 = nn.Conv2d(32 + 10, num_ship_actions + num_shipyard_actions, 1)

        # Value-prediction layers:
        # 2 for the current score_diff and steps_remaining.
        # 10 for the values from get_summary_scalars(...).
        self._state_val_linear = nn.Linear(2 + 10, 1) 

    def forward(self, input_x):
        # TODO: should we convert NCHW back to NHWC at the end?
        x = input_x.permute(0, 3, 1, 2) # from NHWC to NCHW

        # Prepare scalars to be concatenated before final layer.
        summary_scalars = self.get_summary_scalars(x)

        summary_scalars_map = torch.unsqueeze(summary_scalars, -1)
        summary_scalars_map = torch.unsqueeze(summary_scalars_map, -1) # NxCx1x1
        h, w = self._input_hw
        summary_scalars_map = summary_scalars_map.repeat(1, 1, h, w)

        # Actor:
        x_action = x
        x_action = F.relu(self._action_conv1(x_action))
        x_action = F.relu(self._action_conv2(x_action))
        x_action = F.relu(self._action_conv3(x_action))
        x_action = torch.cat((x_action, summary_scalars_map), 1) # Cat along channel dimension.
        action_logits = self._action_conv4(x_action)
        ship_act_logits = action_logits[:, :self._num_ship_actions, :, :]
        shipyard_act_logits = action_logits[:, self._num_ship_actions:, :, :]

        # Critic:
        score_diff_list = input_x[:, 0, 0, 6:7] - input_x[:, 0, 0, 7:8] # Shape: N, 1
        score_diff = score_diff_list[:, 0]
        steps_remaining_list = input_x[:, 0, 0, 8:9] / 400 # TODO: normalization should not be happening here

        x_value = torch.cat((score_diff_list, steps_remaining_list, summary_scalars), dim=1)
        value_residuals = self._state_val_linear(x_value)[:, 0] # Resulting dimension of N.
        value_preds = score_diff + value_residuals

        return ship_act_logits, shipyard_act_logits, value_preds

    def get_summary_scalars(self, x):
        """Extracts key summary pieces of information useful in making both action and value predictions, to be fed into
        the final layers of the two model heads.
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

    def apply_action_distribution(self, action_logits):
        # Assumes that action_logits has shape [B, C, H, W]
        action_logits = action_logits.permute(0, 2, 3, 1) # from BCHW to BHWC

        dist = torch.distributions.categorical.Categorical(logits=action_logits)
        #dist = torch.distributions.independent.Independent(dist, reinterpreted_batch_ndims=2)
        return dist

    def action_log_prob(
        self,
        ship_action_dist,
        shipyard_action_dist,
        state,
        ship_action,
        shipyard_action,
    ):
        """Calculate the log prob of a batch of actions.

        Parameters:
        -----------
        ship_action_dist, shipyard_action_dist : torch.distributions.categorical.Categorical
            Distributions obtained by calling self.apply_action_distribution(...).
        state : Tensor (B,H,W,C)
            State tensor - same one passed to forward(...). This is used to mask out locations where actions won't be
            applied (and this shouldn't be included in log prob calculation).
        ship_action, shipyard_action : Tensor
            Actions sampled from ship_action_dist, shipyard_action_dist.
        """
        # Calculate log prob of actions selected at each individual location.
        ship_action_log_probs = ship_action_dist.log_prob(ship_action) # Shape: (B, H, W)
        shipyard_action_log_probs = shipyard_action_dist.log_prob(shipyard_action) # Shape: (B, H, W)

        # Mask locations that do not have a ship/shipyard as those action selections are irrelevant.
        # Taking sum in log prob space is equivalent to multiplying independent probabilities.
        ship_action_log_prob = torch.sum(ship_action_log_probs * state[:, :, :, 1], dim=(1, 2))
        shipyard_action_log_prob = torch.sum(shipyard_action_log_probs * state[:, :, :, 2], dim=(1, 2))

        action_log_prob = ship_action_log_prob + shipyard_action_log_prob
        return action_log_prob


# class HaliteActorCriticCNN(nn.Module):
#     """CNN that takes as input a Halite state representation and produces:
#     1) a value prediction of expected future rewards under the current policy (critic), and
#     2) a probability distribution over discrete actions at every board position (actor).

#     The actor and critic share the base layers of the model.
#     """
#     # Ideas to try:
#     #   - a few conv with large kernel: 21x21, followed by 1d conv for final prediction
#     #   - try adding batch norm layers
#     #   - add pooling
#     #   - skip connection with raw inputs right before applying final 1d conv operation (particularly if applying pooling)
#     #   - Another hidden layer before value prediction.
#     #   - Reduce dimenionality further before linear layer with another conv, or pooling.
#     #   - ***** both value-prediction and action prediction could probably benefit from feeding some summary stats into the final layer:
#     #       - number of home ships, away ships, home shipyards, away shipyards
#     #       - total amount of halite mined, total amount of halite on ships
#     #       - number of steps remaining
#     def __init__(self, num_actions, input_hw):
#         super().__init__()
#         # Action-probability prediction branch:
#         # Note: padding is set to maintain HxW dimensions throughout.
#         self._action_conv1 = nn.Conv2d(9, 16, 5, padding=2, padding_mode="circular")
#         self._action_conv2 = nn.Conv2d(16, 16, 7, padding=3, padding_mode="circular")
#         self._action_conv3 = nn.Conv2d(16, 32, 7, padding=3, padding_mode="circular")
#         self._action_conv4 = nn.Conv2d(32, 32, 7, padding=3, padding_mode="circular")
#         self._action_conv5 = nn.Conv2d(32, num_actions, 1)

#         # Value-prediction layers:
#         self._state_val_conv1 = nn.Conv2d(9, 16, 7, padding=3, stride=2, padding_mode="circular")
#         h, w = input_hw
#         h = int(np.ceil(h / 2))
#         w = int(np.ceil(w / 2))
#         self._state_val_conv2 = nn.Conv2d(16, 8, 7, padding=3, stride=2, padding_mode="circular")
#         h = int(np.ceil(h / 2))
#         w = int(np.ceil(w / 2))
#         self._state_val_linear = nn.Linear(h * w * 8 + 2, 1) # +2 for the current score_diff and steps_remaining

#         # Shared:
#         self._dropout = nn.Dropout(0.5)

#     def forward(self, input_x):
#         # TODO: should convert NCHW back to NHWC at the end.
#         x = input_x.permute(0, 3, 1, 2) # from NHWC to NCHW

#         # Actor:
#         x_action = x
#         x_action = F.relu(self._action_conv1(x_action))
#         x_action = self._dropout(x_action)
#         x_action = F.relu(self._action_conv2(x_action))
#         x_action = self._dropout(x_action)
#         x_action = F.relu(self._action_conv3(x_action))
#         x_action = self._dropout(x_action)
#         x_action = F.relu(self._action_conv4(x_action))
#         x_action = self._dropout(x_action)
#         action_logits = self._action_conv5(x_action)

#         # Critic:
#         # TODO: Shouldn't make assumptions about input_x structure.
#         score_diff_list = input_x[:, 0, 0, 6:7] - input_x[:, 0, 0, 7:8] # Shape: N, 1
#         score_diff = score_diff_list[:, 0]
#         steps_remaining_list = input_x[:, 0, 0, 8:9] / 400 # TODO: normalization should not be happening here

#         x_value = x
#         x_value = F.relu(self._state_val_conv1(x_value))
#         x_value = self._dropout(x_value)
#         x_value = F.relu(self._state_val_conv2(x_value))
#         x_value = self._dropout(x_value)
#         x_value = torch.flatten(x_value, start_dim=1) # start_dim=1 to preserve batch dimension.
#         x_value = torch.cat((x_value, score_diff_list, steps_remaining_list), dim=1)
#         value_residuals = self._state_val_linear(x_value)[:, 0] # Resulting dimension of N.
#         value_preds = score_diff + value_residuals

#         return action_logits, value_preds, value_residuals, score_diff, steps_remaining_list[:, 0]
