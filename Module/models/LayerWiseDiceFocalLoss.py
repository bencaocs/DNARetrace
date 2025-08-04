import torch
import torch.nn as nn
from torch import Tensor

class LayerWiseDiceBalancedCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, smooth: float = 1e-6, reduction: str = "mean"):
        super(LayerWiseDiceBalancedCrossEntropyLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Compute Balanced Cross-Entropy Loss
        bce_loss = self.bce_loss(input, target)

        # Compute Dice Loss
        preds = torch.sigmoid(input)
        preds = preds.view(-1)
        target_expanded = target.view(-1)

        intersection = (preds * target_expanded).sum()
        dice_score = (2. * intersection + self.smooth) / (preds.sum() + target_expanded.sum() + self.smooth)
        dice_loss = 1 - dice_score

        w1 = 1.2
        w2 = 0.8
        return w1 * bce_loss + w2 * dice_loss