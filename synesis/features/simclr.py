from pathlib import Path

import torch
import wget
from torch import nn
from torchvision.models import resnet50


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR(nn.Module):
    def __init__(self, feature_extractor=True):
        super(SimCLR, self).__init__()

        self.feature_extractor = feature_extractor

        if not Path("models/pretrained/simclr.pt").exists():
            # download from github
            print("Downloading pretrained weights for SimCLR")
            wget.download(
                "https://github.com/Spijkervet/SimCLR/releases/download/1.2/checkpoint_100.tar",
            )
            Path("checkpoint_100.tar").rename("models/pretrained/simclr.pt")

        self.encoder = resnet50(pretrained=False)
        self.n_features = self.encoder.fc.in_features

        self.encoder.fc = Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, 64, bias=False),
        )

    def forward(self, x):
        if self.feature_extractor:
            with torch.no_grad():
                h = self.encoder(x)
                return h
        else:
            raise NotImplementedError("Training not implemented yet.")
