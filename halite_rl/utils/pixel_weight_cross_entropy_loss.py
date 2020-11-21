import torch

class PixelWeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Extends torch.nn.CrossEntropyLoss and adds support for 
    pixelwise multiplication of losses with weight map before
    reduction.
    """

    def __init__(self, weight):
        super().__init__(weight=weight, reduction='none')

    def forward(self, input, target, pixel_weights):
        pixel_ce_loss = super().forward(input, target)
        weighted_pixel_ce_loss = pixel_ce_loss * pixel_weights
        return torch.mean(weighted_pixel_ce_loss)
