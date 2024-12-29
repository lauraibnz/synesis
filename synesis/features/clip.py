import torch
from torch import nn
from transformers import CLIPModel


class CLIP(nn.Module):
    def __init__(self, feature_extractor=True):
        super(CLIP, self).__init__()

        self.feature_extractor = feature_extractor

        # Load the pretrained CLIP model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, x):
        if self.feature_extractor:
            with torch.no_grad():
                outputs = self.model.get_image_features(pixel_values=x)
                return outputs
        else:
            raise NotImplementedError("Training not implemented yet.")
