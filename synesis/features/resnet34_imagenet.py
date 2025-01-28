"""ResNet model for feature extraction."""

import torch
from torch import nn
from torchvision.models import resnet34


class ResNet34_ImageNet(nn.Module):
    def __init__(self, feature_extractor=True):
        super(ResNet34_ImageNet, self).__init__()

        self.feature_extractor = feature_extractor

        self.encoder = resnet34(weights="IMAGENET1K_V1")
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        if self.feature_extractor:
            with torch.no_grad():
                h = self.encoder(x)
                return h
        else:
            raise NotImplementedError("Training not implemented yet.")
