"""CodiCodec latent feature extractor.

This wrapper always flattens non-temporal CodiCodec axes into one feature axis.
For latents shaped like [time, axis_2, axis_3], this produces [time, feature].
It can then either mean-pool over time or return [feature, time].
"""

import torch
from torch import nn

from codicodec import EncoderDecoder


class CodiCodec(nn.Module):
    def __init__(self, feature_extractor=False, extract_kws=None, **kwargs):
        super(CodiCodec, self).__init__()

        del kwargs
        self.encoder = EncoderDecoder()
        self.feature_extractor = feature_extractor
        self.extract_kws = extract_kws or {}

        print("CodiCodec model initialized")
        print(self.encoder)

    def _flatten_non_temporal_axes(self, latents: torch.Tensor) -> torch.Tensor:
        """Flatten all non-temporal latent axes into one feature axis."""
        if latents.dim() > 3 and latents.shape[0] == 1:
            latents = latents.squeeze(0)

        if latents.dim() == 1:
            return latents.unsqueeze(0)

        if latents.dim() == 2:
            return latents

        if latents.dim() != 3:
            raise ValueError(
                "Expected CodiCodec latents to have shape [time, axis_2, axis_3]. "
                f"Got shape {tuple(latents.shape)}."
            )

        return latents.reshape(latents.shape[0], -1)

    @torch.no_grad()
    def extract_features(self, x, pool_time=True):
        features = []

        for sample in x:
            latent = self.encoder.encode(sample.detach().cpu().float().numpy())
            latent = torch.as_tensor(latent, dtype=torch.float32)
            latent = self._flatten_non_temporal_axes(latent)
            if pool_time:
                latent = latent.mean(dim=0)
            else:
                latent = latent.transpose(0, 1)
            features.append(latent)

        return {"latents": torch.stack(features, dim=0)}

    def forward(self, x, pool_time=True):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            x = x.mean(dim=1)

        if self.extract_kws:
            pool_time = self.extract_kws.get("pool_time", pool_time)

        return self.extract_features(x, pool_time=pool_time)["latents"]
