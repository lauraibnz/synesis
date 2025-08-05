"""
    Mel spectrogram feature extraction.
"""
import torch
from torchaudio.transforms import MelSpectrogram

class MelSpec(MelSpectrogram):
    def __init__(self, feature_extractor=None, \
                 extract_kws = {}, **kwargs):
        
        self.extract_kws = extract_kws
        self.feature_extractor = feature_extractor
    
        sample_rate = kwargs.get('sample_rate', 16000)
        n_mels = self.extract_kws.get('n_mels', 256)
        n_fft = self.extract_kws.get('n_fft', 2048)
        win_length = self.extract_kws.get('win_length', 2048)
        hop_length = self.extract_kws.get('hop_length', 512)

        super().__init__(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, 
                         win_length=win_length, hop_length=hop_length, f_min=30, f_max=sample_rate // 2)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape (channels, time) or (time,) representing the audio signal.
        
        Returns:
            Tensor of shape (n_mels, time) representing the mel spectrogram.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel_spec = super().forward(waveform)

        # find the squared absolute mel spectrogram
        mel_spec = torch.abs(mel_spec)**2

        # clip to avoid log(0)
        mel_spec = torch.clamp(mel_spec, min=1e-10)

        # Get the log-mel spectrogram
        log_mel_spec = torch.log1p(mel_spec)
        return log_mel_spec