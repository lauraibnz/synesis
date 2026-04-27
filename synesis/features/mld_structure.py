import gin
import torch
from torch import nn

import mld.pipeline.model
import mld.pipeline.networks


class MLD_Structure(nn.Module):
    def __init__(self, feature_extractor=True, extract_kws={}, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_batch_paths = None
        self.feature_name = kwargs.get("feature_name")

        if not extract_kws:
            raise ValueError("Model/config paths must be provided.")

        checkpoint_path = extract_kws["checkpoint_path"]
        config_path = extract_kws["config_path"]
        use_ema = extract_kws.get("use_ema", True)

        gin.clear_config()
        gin.parse_config_file(config_path)

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

        self.latent_codec = self.model.build_latent_codec(device=self.device)
        if self.latent_codec is None:
            raise ValueError("Could not build latent codec from the loaded model.")

    def set_batch_paths(self, batch_paths):
        self.current_batch_paths = [str(path) for path in batch_paths]

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

        z = self.model.encode_audio_batch_to_model_latent(
            x,
            codec=self.latent_codec,
            device=self.device,
        )
        z = self.model._normalize_latents(z)
        structure_emb = self.model.encode_structure(z)

        self.current_batch_paths = None
        return structure_emb.cpu()
