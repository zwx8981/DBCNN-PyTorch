import torch
import torch.nn as nn
from torch.autograd import Variable


class EDMLoss(nn.Module):
    def __init__(self):
        """
        Initialize the state of the class

        Args:
            self: (todo): write your description
        """
        super(EDMLoss, self).__init__()

    def forward(self, p_target: Variable, p_estimate: Variable):
        """
        R calculate the mean probability.

        Args:
            self: (todo): write your description
            p_target: (todo): write your description
            p_estimate: (todo): write your description
        """
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()