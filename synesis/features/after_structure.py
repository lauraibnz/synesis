import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

# Add AFTER codebase to sys.path
AFTER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../AFTER'))
if AFTER_PATH not in sys.path:
    sys.path.append(AFTER_PATH)

from after.diffusion import RectifiedFlow

class AFTER_Structure(nn.Module):
    def __init__(self, feature_extractor=True):
        super(AFTER_Structure, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "../../../AFTER/experiments/after_runs/slakh2100_train/")
        step = 800000
        autoencoder_path = os.path.join(BASE_DIR, "../../../AFTER/pretrained/AE_slakh.pt")
        checkpoint_path = os.path.join(model_path, f"checkpoint{step}_EMA.pt")
        config_path = os.path.join(model_path, "config.gin")

        import gin
        gin.enter_interactive_mode()
        gin.parse_config_file(config_path)

        # Load model
        self.model = RectifiedFlow(device=self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)["model_state"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.eval().to(self.device)

        # Load autoencoder
        self.emb_model = torch.jit.load(autoencoder_path).eval().to(self.device)
        self.model.emb_model = self.emb_model

    @torch.no_grad()
    def forward(self, x):
        # x: (batch, channels, samples) or (batch, samples)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.to(self.device)
        # Encode waveform to latent
        z = self.emb_model.encode(x)
        # Extract structure embedding
        structure_emb = self.model.encoder_time(z)  # [batch, 12, time]
        return structure_emb.cpu()