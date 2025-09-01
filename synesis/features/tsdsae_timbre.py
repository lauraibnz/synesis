import os
import sys
import torch
from torch import nn
import torchaudio
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'externals', 'dSEQ-VAE')
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.ts_dsae import TsDsae
from src.datasets.constants.slakh import SR, NFFT, HOP, NMEL


class TSDSAE_Timbre(nn.Module):
    def __init__(self, feature_extractor=True, extract_kws={}):
        super(TSDSAE_Timbre, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if extract_kws == {}:
            raise ValueError("Model checkpoint path must be provided in extract_kws.")
        
        checkpoint_path = extract_kws.get("checkpoint_path")
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided in extract_kws")

        # Load model
        self.model = TsDsae.load_from_checkpoint(checkpoint_path, strict=False)
        self.model = self.model.eval().to(self.device)

        # Setup preprocessing transforms
        # Use the original HOP constant that the model was trained with
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, 
            n_fft=NFFT, 
            hop_length=HOP,
            n_mels=NMEL
        ).to(self.device)
        
        self.log_compress = lambda x: torch.log(torch.finfo(x.dtype).eps + x)

    @torch.no_grad()
    def forward(self, x):
        """
        Extract timbre embeddings (global embeddings) from audio
        x: (batch, channels, samples) or (batch, samples) - already processed to 4-second chunks
        Returns: timbre_emb [batch, T, z_dim] - timbre embeddings
        """
        x = x.to(self.device)
        
        # x is already in the correct format from the dataset pipeline
        # [batch, channels, samples] where each sample is 4 seconds
        
        # Convert to mono if stereo
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Normalize
        x = x / (torch.max(torch.abs(x)) + 1e-8)
        
        # Apply mel spectrogram
        mel_spec = self.mel(x)  # [batch, NMEL, T]
        
        # Remove extra channel dimension if present
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)  # Remove channel dimension
        
        # Apply log compression
        mel_log = self.log_compress(mel_spec)  # [batch, NMEL, T]
        
        # Ensure all chunks have exactly 251 time steps
        if mel_log.shape[2] != 251:
            if mel_log.shape[2] > 251:
                # Truncate if too long
                mel_log = mel_log[:, :, :251]
            else:
                # Pad if too short (shouldn't happen with 4-second chunks)
                padding = 251 - mel_log.shape[2]
                mel_log = F.pad(mel_log, (0, padding))
        
        # Transpose for model input [batch, T, NMEL]
        mel_input = mel_log.transpose(1, 2)  # [batch, T, NMEL]
        
        # Create sequence lengths (all chunks are the same length)
        seq_lengths = torch.tensor([mel_input.shape[1]] * mel_input.shape[0], dtype=torch.long, device=self.device)
        
        # Forward pass through model
        outputs = self.model(mel_input, seq_lengths, deterministic=True)
        
        # Extract timbre embeddings (global) [batch, T, z_dim]
        timbre_emb = outputs['v']  # [batch, T, z_dim]

        return timbre_emb.cpu()