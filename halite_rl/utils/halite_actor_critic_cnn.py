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
    def __init__(self, num_actions, input_hw):
        super().__init__()
        # Action-probability prediction branch:
        # Note: padding is set to maintain HxW dimensions throughout.
        self._action_conv1 = nn.Conv2d(9, 16, 5, padding=2, padding_mode="circular")
        self._action_conv2 = nn.Conv2d(16, 16, 7, padding=3, padding_mode="circular")
        self._action_conv3 = nn.Conv2d(16, 32, 7, padding=3, padding_mode="circular")
        self._action_conv4 = nn.Conv2d(32, num_actions, 1)

        # Value-prediction layers:
        self._state_val_linear = nn.Linear(2, 1) # 2 for the current score_diff and steps_remaining

    def forward(self, input_x):
        # TODO: should convert NCHW back to NHWC at the end.
        x = input_x.permute(0, 3, 1, 2) # from NHWC to NCHW

        # Actor:
        x_action = x
        x_action = F.relu(self._action_conv1(x_action))
        x_action = F.relu(self._action_conv2(x_action))
        x_action = F.relu(self._action_conv3(x_action))
        action_logits = self._action_conv4(x_action)

        # Critic:
        # TODO: Shouldn't make assumptions about input_x structure.
        score_diff_list = input_x[:, 0, 0, 6:7] - input_x[:, 0, 0, 7:8] # Shape: N, 1
        score_diff = score_diff_list[:, 0]
        steps_remaining_list = input_x[:, 0, 0, 8:9] / 400 # TODO: normalization should not be happening here

        x_value = torch.cat((score_diff_list, steps_remaining_list), dim=1)
        value_residuals = self._state_val_linear(x_value)[:, 0] # Resulting dimension of N.
        value_preds = score_diff + value_residuals

        # HACK: shouldn't be returning all these values.
        return action_logits, value_preds, value_residuals, score_diff, steps_remaining_list[:, 0]

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
