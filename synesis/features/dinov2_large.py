"""DINOv2.
https://github.com/facebookresearch/dinov2
License: Apache-2.0 (see NOTICE for full license)
"""

import torch
from torch import nn
from transformers import AutoModel


class DINOv2_large(nn.Module):
    def __init__(self, feature_extractor=True):
        super(DINOv2_large, self).__init__()

        self.feature_extractor = feature_extractor

        self.model = AutoModel.from_pretrained("facebook/dinov2-large")

    def forward(self, x):
        if self.feature_extractor:
            with torch.no_grad():
                h = self.model(x).last_hidden_state
                h = h.mean(dim=[-2])
                return h
        else:
            raise NotImplementedError("Training not implemented yet.")
