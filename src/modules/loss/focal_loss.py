import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for classification tasks.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        """
        :param alpha: Weighting factor for classes. Scalar or tensor.
        :param gamma: Focusing parameter.
        :param reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # compute log probabilities
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        # gather log probabilities of the true class
        logpt = logpt.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        loss = -self.alpha * ((1 - pt) ** self.gamma) * logpt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
