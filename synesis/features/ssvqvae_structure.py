import os
import sys
import torch
from torch import nn
import confugue
import librosa
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'externals', 'ss-vq-vae', 'src')
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ss_vq_vae.models.vqvae_oneshot import Experiment

class SSVQVAE_Structure(nn.Module):
    def __init__(self, feature_extractor=True, extract_kws={}):
        super(SSVQVAE_Structure, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if extract_kws == {}:
            raise ValueError("Model/config paths must be provided.")
        
        logdir = extract_kws.get("logdir")
        model_state_path = extract_kws.get("model_state_path")
        config_path = extract_kws.get("config_path")

        # Load configuration and model
        cfg = confugue.Configuration.from_yaml_file(config_path)
        self.exp = cfg.configure(Experiment, logdir=logdir, device=self.device)
        
        # Load model state
        state_dict = torch.load(model_state_path, map_location=self.device)
        self.exp.model.load_state_dict(state_dict)
        self.exp.model = self.exp.model.eval().to(self.device)

    @torch.no_grad()
    def forward(self, x):
        # x: (batch, channels, samples) or (batch, samples)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.to(self.device)
        
        # Preprocess audio to spectrogram
        if x.shape[1] == 1:  # If mono audio
            x = x.squeeze(1)  # Remove channel dimension for librosa
            x = x.cpu().numpy()
            # Apply the same preprocessing as Experiment
            spec = self.exp.preprocess(x)
            x = torch.as_tensor(spec, device=self.device)
        else:
            # If already spectrogram, just ensure correct format
            x = x.to(self.device)
        
        # Extract content embedding (continuous quantized embeddings)
        encoded_c, discrete_ids, losses_c = self.exp.model.encode_content(x)
        
        # Return continuous quantized embeddings
        # encoded_c shape: [batch, 1024, time] - continuous vectors from codebook
        return encoded_c.cpu()
