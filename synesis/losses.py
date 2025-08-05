"""
    File containing custom loss functions for the Synesis project.
"""
import torch

class MaskedBce(torch.nn.Module):
    """
        Binary crossentropy (BCE) with masking. The positions where mask is 0 will be 
        deactivated when calculating the loss.
    """
    def __init__(self):
        super(MaskedBce, self).__init__()

    def forward(self, output, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target, dtype=torch.float32)
        epsilon = 1e-7
        output = torch.clamp(output, epsilon, 1. - epsilon)
        standard_bce = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
        loss = torch.sum(standard_bce * mask) / torch.sum(mask)
        return loss