import gin
import torch
from torch import nn
from pathlib import Path

import mld.pipeline.model      # Register gin-configurable model classes.
import mld.pipeline.networks   # Register gin-configurable network classes.


class MLD_Timbre(nn.Module):
    def __init__(self, feature_extractor=True, extract_kws={}, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_name = kwargs.get("feature_name")
        self.current_batch_paths = None

        if not extract_kws:
            raise ValueError("Model/config paths must be provided.")

        checkpoint_path = Path(extract_kws["checkpoint_path"])
        config_path = Path(extract_kws["config_path"])
        use_ema = extract_kws.get("use_ema", True)

        gin.clear_config()

        # Be tolerant to adapted/older saved configs when possible.
        config_text = config_path.read_text()
        config_text = config_text.replace("musdis2.", "mld.pipeline.model.")
        config_text = config_text.replace("musdis3.", "mld.pipeline.networks.")
        config_text = config_text.replace("musdis4.", "mld.pipeline.utils.")
        config_text = config_text.replace("musdis.", "mld.")
        gin.parse_config(config_text)

        model_class_ref = gin.query_parameter("%model_class")
        model_class = model_class_ref.configurable.wrapped
        self.model = model_class().to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if use_ema and "ema_state_dict" in ckpt:
            print("Loading EMA weights for stable feature extraction.")
            sd = self.model.state_dict()
            for k, v in ckpt["ema_state_dict"].items():
                if k in sd and sd[k].shape == v.shape:
                    sd[k] = v
            self.model.load_state_dict(sd, strict=False)
        else:
            print("Loading regular model weights.")
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)

        self.model.eval().to(self.device)

        self.timbre_input_key = str(getattr(self.model, "timbre_input_key", "latent"))
        self.latent_codec_name = getattr(self.model, "latent_codec", None)

        if self.timbre_input_key not in {"latent", "latent_mean", "latent_mean_std"}:
            raise NotImplementedError(
                f"Unsupported timbre_input_key for clean MLD extractor: {self.timbre_input_key}"
            )

        self.latent_codec = self.model.build_latent_codec(device=self.device)
        if self.latent_codec is None:
            raise ValueError(
                "Model uses latent-based timbre extraction but no latent codec was recorded."
            )

        print(f"Loaded latent codec from model metadata: {self.latent_codec_name}")

    def set_batch_paths(self, batch_paths):
        self.current_batch_paths = [str(path) for path in batch_paths]

    def _cache_path_for_feature_path(self, feature_path: str) -> Path:
        path = Path(feature_path)

        if self.feature_name and self.feature_name in path.parts:
            idx = path.parts.index(self.feature_name)
            base = Path(*path.parts[:idx])
            suffix = Path(*path.parts[idx + 1 :])
            return base / "_latent_cache" / str(self.latent_codec_name) / suffix

        return path.parent.parent / "_latent_cache" / str(self.latent_codec_name) / path.name

    def _load_cached_latent(self, cache_path: Path) -> torch.Tensor:
        latent = torch.load(cache_path, map_location="cpu", weights_only=False)
        if latent.dim() == 3 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        return latent.to(dtype=torch.float32)

    def _encode_or_load_latent_batch(self, x: torch.Tensor) -> torch.Tensor:
        if not self.current_batch_paths:
            return self.model.encode_audio_batch_to_model_latent(
                x,
                codec=self.latent_codec,
                device=self.device,
            )

        cache_paths = [self._cache_path_for_feature_path(path) for path in self.current_batch_paths]
        latents = [None] * len(cache_paths)
        missing_indices = []

        for idx, cache_path in enumerate(cache_paths):
            if cache_path.exists():
                latents[idx] = self._load_cached_latent(cache_path)
            else:
                missing_indices.append(idx)

        if missing_indices:
            missing_waveforms = x[missing_indices]
            missing_latents = self.model.encode_audio_batch_to_model_latent(
                missing_waveforms,
                codec=self.latent_codec,
                device=self.device,
            ).detach().cpu()

            for offset, idx in enumerate(missing_indices):
                cache_path = cache_paths[idx]
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                latent = missing_latents[offset]
                torch.save(latent, cache_path)
                latents[idx] = latent.to(dtype=torch.float32)

        return torch.stack(latents, dim=0).to(self.device, dtype=torch.float32)

    def _latents_to_timbre_source(self, z: torch.Tensor) -> torch.Tensor:
        if self.timbre_input_key == "latent":
            return z
        if self.timbre_input_key == "latent_mean":
            return z.mean(dim=-1)
        if self.timbre_input_key == "latent_mean_std":
            return torch.cat([z.mean(dim=-1), z.std(dim=-1)], dim=1)
        raise RuntimeError(f"Unhandled timbre_input_key: {self.timbre_input_key}")

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)

        # Accept [B, T] or [B, C, T]
        if x.dim() == 3:
            x = x.mean(dim=1)

        if x.dim() != 2:
            raise ValueError(
                f"Expected waveform batch [B, T] or [B, C, T], got {tuple(x.shape)}"
            )

        z = self._encode_or_load_latent_batch(x)
        z = self.model._normalize_latents(z)
        self.current_batch_paths = None

        timbre_source = self._latents_to_timbre_source(z)
        timbre_emb = self.model.encode_timbre(timbre_source)
        return timbre_emb.cpu()
