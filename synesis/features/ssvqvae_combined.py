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

class SSVQVAE_Combined(nn.Module):
    def __init__(self, feature_extractor=True, extract_kws={}):
        super(SSVQVAE_Combined, self).__init__()
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
        l_style = torch.full((x.shape[0],), x.shape[2], device=self.device)

        # Extract style embedding
        encoded_s, losses_s = self.exp.model.encode_style(x, l_style)
        encoded_c, discrete_ids, losses_c = self.exp.model.encode_content(x)
        
        # Style Conditioning Approach: Use style to modulate content rather than concatenate
        # This reduces overfitting to style-specific characteristics while preserving structure
        
        # Expand style to match content temporal dimension
        encoded_s_expanded = encoded_s.unsqueeze(2).repeat(1, 1, encoded_c.shape[2])
        
        # Option A: Simple weighted combination (reduces dimensionality)
        #style_weight = 0.1  # Small influence to preserve content structure
        #combined_emb = encoded_c + style_weight * encoded_s_expanded
        
        # Option B: Multiplicative conditioning (uncomment to use instead)
        # style_gates = torch.sigmoid(encoded_s_expanded)  # Gate values 0-1
        # combined_emb = encoded_c * (1 + 0.1 * style_gates)  # Small style modulation
        
        # Option C: Original concatenation (comment out A or B to use)
        combined_emb = torch.cat((encoded_c, encoded_s_expanded), dim=1)
        
        return combined_emb.cpu()
