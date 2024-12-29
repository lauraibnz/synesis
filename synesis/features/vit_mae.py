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
                # 32, 768 to 768
                h = h.mean(dim=0)
                return h
        else:
            raise NotImplementedError("Training not implemented yet.")
