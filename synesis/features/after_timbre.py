import os
import sys
import torch
from torch import nn
from after.diffusion import RectifiedFlow
import gin

class AFTER_Timbre(nn.Module):
    def __init__(self, feature_extractor=True, extract_kws={}):
        super(AFTER_Timbre, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if extract_kws == {}:
            raise ValueError("Model/config paths must be provided.")
        
        autoencoder_path = extract_kws.get("autoencoder_path")
        checkpoint_path = extract_kws.get("checkpoint_path")
        config_path = extract_kws.get("config_path")
        
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
        # Extract timbre embedding
        timbre_emb = self.model.encoder(z)
        return timbre_emb.cpu()
