"""Wav2Vec2.
https://docs.pytorch.org/audio/main/generated/torchaudio.models.Wav2Vec2Model.html
License: BSD-2-Clause
"""

import torch
import torchaudio
from torch import nn


class Wav2Vec2(nn.Module):
    def __init__(self, feature_extractor=True):
        super(Wav2Vec2, self).__init__()

        self.feature_extractor = feature_extractor

        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model()

    def forward(self, x):
        # squeeze channel dim
        x = x.squeeze(1)
        with torch.inference_mode():
            # Extract features and clone/detach immediately
            features = self.model.extract_features(x)[0][-1].clone().detach()

        features = features.mean(dim=1)
        return features
