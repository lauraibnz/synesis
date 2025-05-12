"""I-JEPA.
https://github.com/facebookresearch/ijepa
License: Attribution-NonCommercial 4.0 International (see NOTICE for full license)
"""

import torch
from torch import nn
from transformers import AutoModel


class IJEPA(nn.Module):
    def __init__(self, feature_extractor=True):
        super(IJEPA, self).__init__()

        self.feature_extractor = feature_extractor

        self.encoder = AutoModel.from_pretrained("jmtzt/ijepa_vith14_1k")

    def forward(self, x):
        if self.feature_extractor:
            with torch.no_grad():
                h = self.encoder(pixel_values=x).last_hidden_state.mean(dim=1)
                return h
        else:
            raise NotImplementedError("Training not implemented yet.")
