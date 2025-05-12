"""Music2Latent.
https://github.com/SonyCSLParis/music2latent
License: CC Attribution-NonCommercial 4.0 International (see NOTICE for full license)
"""

from music2latent import EncoderDecoder
from torch import nn
import torch


class MusicLatent(nn.Module):
    def __init__(self, feature_extractor=False, extract_kws={}, **kwargs):
        super(MusicLatent, self).__init__()

        self.encoder = EncoderDecoder()
        self.gen = self.encoder.gen
        self.feature_extractor = feature_extractor  # different than extract_features, which is generation vs analysis
        self.extract_kws = extract_kws

        print("Music2Latent model initialized")
        print(self.encoder)

    def freeze(self):
        for param in self.encoder.gen.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_features(self, x, pool_hop=-1, extract_features=True):
        latents = self.encoder.encode(x, extract_features=extract_features).permute(
            0, 2, 1
        )

        ## average the latents with average pooling with a hop of pool_hop
        if pool_hop == -1:
            return {"latents": latents.mean(dim=1)}
        split_latents = torch.split(latents, pool_hop, dim=1)
        averaged_latents = torch.stack(
            [torch.mean(latent, dim=1) for latent in split_latents], dim=1
        )
        averaged_latents = averaged_latents.squeeze(1)

        return {"latents": averaged_latents}

    def forward(self, x, pool_hop=-1, extract_features=True):
        if x.dim() == 3:  # stereo
            x = x.mean(dim=1)

        # override kws with self.extract_kws if they are changed
        if self.extract_kws:
            extract_features = self.extract_kws.get(
                "extract_features", extract_features
            )
            pool_hop = self.extract_kws.get("pool_hop", pool_hop)

        return self.extract_features(x, pool_hop, extract_features=extract_features)[
            "latents"
        ]
