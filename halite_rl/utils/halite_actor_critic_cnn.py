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
    def __init__(self, num_actions, input_hw):
        super().__init__()
        # Base layers (shared):
        # Note: padding is set to maintain HxW dimensions throughout.
        self._base_conv1 = nn.Conv2d(9, 16, 7, padding=3, padding_mode="circular")
        self._base_conv2 = nn.Conv2d(16, 16, 7, padding=3, padding_mode="circular")
        self._base_conv3 = nn.Conv2d(16, 32, 21, padding=10, padding_mode="circular") # Sees entire board

        # Action-probability prediction layer:
        self._action_prob_conv1 = nn.Conv2d(32, 32, 21, padding=10, padding_mode="circular")
        self._action_prob_conv2 = nn.Conv2d(32, num_actions, 1)

        # Value-prediction layers:
        # Main purposed of conv5 is to reduce dimensionality before linear layer.
        channels = 8
        downsample_stride = 2
        h, w = input_hw
        self._state_val_conv1 = nn.Conv2d(32, channels, 7, padding=3, stride=downsample_stride, padding_mode="circular")
        h = int(np.ceil(h / downsample_stride))
        w = int(np.ceil(w / downsample_stride))
        self._state_val_conv2 = nn.Conv2d(channels, channels, 7, padding=3, stride=downsample_stride, padding_mode="circular")
        h = int(np.ceil(h / downsample_stride))
        w = int(np.ceil(w / downsample_stride))
        linear_in_len = h * w * channels + 2 # +2 for the score_diff and steps remaining terms
        self._value_linear1 = nn.Linear(linear_in_len, 1)

        # Shared:
        self._dropout = nn.Dropout(0.25)

    def forward(self, input_x):
        # TODO: should convert NCHW back to NHWC at the end.

        # Shared base:
        x = input_x.permute(0, 3, 1, 2) # from NHWC to NCHW
        x = F.relu(self._base_conv1(x))
        x = self._dropout(x)
        x = F.relu(self._base_conv2(x))
        x = self._dropout(x)
        x = F.relu(self._base_conv3(x))
        x = self._dropout(x)

        # Actor:
        action_x = self._action_prob_conv1(x)
        action_x = self._dropout(action_x)
        action_logits = self._action_prob_conv2(action_x)

        # Critic:
        # TODO: Shouldn't make assumptions about input_x structure.
        score_diff_list = input_x[:, 0, 0, 6:7] - input_x[:, 0, 0, 7:8] # Shape: N, 1
        score_diff = score_diff_list[:, 0]
        steps_remaining_list = input_x[:, 0, 0, 8:9]

        value_x = F.relu(self._state_val_conv1(x))
        value_x = self._dropout(value_x)
        value_x = F.relu(self._state_val_conv2(value_x))
        value_x = self._dropout(value_x)
        value_x = torch.flatten(value_x, start_dim=1) # start_dim=1 to preserve batch dimension.
        value_x = torch.cat((value_x, score_diff_list, steps_remaining_list), dim=1)
        value_residuals = self._value_linear1(value_x)[:, 0] # Resulting dimension of N.
        value_preds = score_diff + value_residuals
        return action_logits, value_preds
