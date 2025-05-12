"""ResNet.
https://docs.pytorch.org/vision/main/models/resnet.html
License: BSD 3-Clause License (see NOTICE for full license)
"""

import torch
from torch import nn
from torchvision.models import resnet101


class ResNet101_ImageNet(nn.Module):
    def __init__(self, feature_extractor=True):
        super(ResNet101_ImageNet, self).__init__()

        self.feature_extractor = feature_extractor

        self.encoder = resnet101(weights="IMAGENET1K_V2")
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        if self.feature_extractor:
            with torch.no_grad():
                h = self.encoder(x)
                return h
        else:
            raise NotImplementedError("Training not implemented yet.")
