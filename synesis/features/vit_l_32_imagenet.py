"""ViT model for feature extraction."""

import torch
from torch import nn
from torchvision.models import vit_l_32  # or another ViT model varian


class ViT_l_32_ImageNet(nn.Module):
    def __init__(self, feature_extractor=True):
        super(ViT_l_32_ImageNet, self).__init__()

        self.feature_extractor = feature_extractor

        # Load the pretrained ViT model
        self.encoder = vit_l_32(weights="IMAGENET1K_V1")
        self.encoder.heads = (
            nn.Identity()
        )  # Replace the classification head with an identity layer

    def forward(self, x):
        if self.feature_extractor:
            with torch.no_grad():
                h = self.encoder(x)
                return h
        else:
            raise NotImplementedError("Training not implemented yet.")


# Example usage
if __name__ == "__main__":
    model = ViT_l_32_ImageNet()
    print(model)
