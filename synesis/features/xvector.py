"""XVector.
https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
License: Apache 2.0 (see NOTICE for full license)
"""

import torch
from speechbrain.pretrained import EncoderClassifier
from torch import nn


class XVector(nn.Module):
    def __init__(self, feature_extractor=True, device="cuda"):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-xvect-voxceleb", run_opts={"device": self.device}
        )
        self.encoder.eval()
        self.feature_extractor = feature_extractor

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        return self

    def embed(self, x):
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.squeeze(1)
        features = self.encoder.encode_batch(x)
        if features.dim() == 3:
            features = features.squeeze(1)

        return features

    def forward(self, x):
        return self.embed(x)


if __name__ == "__main__":
    import torch

    xvector = XVector()
    x = torch.randn(4, 16000 * 10)
    features = xvector.embed(x)
    print(features.shape)
