"""HuBERT.
https://github.com/bshall/hubert
License: MIT (see NOTICE for full license)
"""

import torch
import torchaudio
from torch import nn


class HuBERT(nn.Module):
    def __init__(self, feature_extractor=True):
        super(HuBERT, self).__init__()

        self.feature_extractor = feature_extractor

        self.bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = self.bundle.get_model()

    def forward(self, x):
        # squeeze channel dim
        x = x.squeeze(1)
        with torch.inference_mode():
            # Extract features and clone/detach immediately
            features = self.model.extract_features(x)[0][-1].clone().detach()

        features = features.mean(dim=1)
        return features
