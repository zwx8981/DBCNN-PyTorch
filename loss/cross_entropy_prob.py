import torch

import torch.nn as nn
import torch.nn.functional as F

class cross_entropy_prob(nn.Module):
    def __init__(self):
        """
        Initialize the entropy.

        Args:
            self: (todo): write your description
        """
        super(cross_entropy_prob, self).__init__()

    def forward(self, pred, soft_targets):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            pred: (todo): write your description
            soft_targets: (todo): write your description
        """
        pred = F.log_softmax(pred)
        loss = torch.mean(torch.sum(- soft_targets * pred, 1))
        return loss
