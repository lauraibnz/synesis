"""ViT MAE.
https://github.com/facebookresearch/mae
https://huggingface.co/timm/vit_huge_patch14_224.mae
License: CC-BY-NC 4.0 (see NOTICE for full license)
"""

import timm
import torch
from torch import nn


class ViT_MAE(nn.Module):
    def __init__(self, feature_extractor=True):
        super(ViT_MAE, self).__init__()

        self.feature_extractor = feature_extractor

        self.encoder = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )

    def forward(self, x):
        if self.feature_extractor:
            with torch.no_grad():
                h = self.encoder(x)
                return h
        else:
            raise NotImplementedError("Training not implemented yet.")
