
import torch.nn as nn
import torch.nn.functional as F

class ImitationCNN(nn.Module):
    # Ideas to try:
    #   - a few conv with large kernel: 21x21, followed by 1d conv for final prediction
    #   - try adding batch norm layers
    #   - add pooling
    #   - skip connection with raw inputs right before applying final 1d conv operation (particularly if applying pooling)
    #   - 
    def __init__(self):
        super().__init__()
        # Note: padding is set to maintain HxW dimensions throughout.
        self._conv1 = nn.Conv2d(9, 16, 5, padding=2, padding_mode="circular")
        self._conv2 = nn.Conv2d(16, 16, 5, padding=2, padding_mode="circular")
        self._conv3 = nn.Conv2d(16, 32, 7, padding=3, padding_mode="circular")
        self._conv4 = nn.Conv2d(32, 8, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # from NHWC to NCHW
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        x = F.relu(self._conv3(x))
        x = self._conv4(x)
        return x
