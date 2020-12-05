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
        self._conv1 = nn.Conv2d(9, 16, 5, padding=2, padding_mode="circular")
        self._conv2 = nn.Conv2d(16, 16, 5, padding=2, padding_mode="circular")
        self._conv3 = nn.Conv2d(16, 32, 7, padding=3, padding_mode="circular")

        # Action-probability prediction layer:
        self._conv4 = nn.Conv2d(32, num_actions, 1)

        # Value-prediction layers:
        # Main purposed of conv5 is to reduce dimensionality before linear layer.
        num_chan_conv5 = 4
        stride_conv5 = 2
        self._conv5 = nn.Conv2d(32, num_chan_conv5, 7, padding=3, stride=stride_conv5, padding_mode="circular")
        h, w = input_hw
        linear_in_len = int(np.ceil(h / stride_conv5)) * int(np.ceil(w / stride_conv5)) * num_chan_conv5 
        self._linear1 = nn.Linear(linear_in_len, 1)

    def forward(self, x):
        # TODO: should convert NCHW back to NHWC at the end.

        # Shared base:
        x = x.permute(0, 3, 1, 2) # from NHWC to NCHW
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        x = F.relu(self._conv3(x))

        # Actor:
        action_logits = self._conv4(x)

        # Critic:
        value_x = F.relu(self._conv5(x))
        value_x = torch.flatten(value_x, start_dim=1) # start_dim=1 to preserve batch dimension.
        value_preds = self._linear1(value_x)
        return action_logits, value_preds
