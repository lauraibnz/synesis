import torch
from einops import rearrange
from torch import nn
from torchaudio.compliance import kaldi
from transformers import AutoModel

# loading our model weights
# loading the corresponding preprocessor config


class AudioMAE(nn.Module):
    def __init__(self, feature_extractor=False, extract_kws={}, **kwargs):
        super(AudioMAE, self).__init__()

        self.model = AutoModel.from_pretrained(
            "hance-ai/audiomae", trust_remote_code=True
        )
        self.feature_extractor = feature_extractor  # different than extract_features, which is generation vs analysis
        self.extract_kws = extract_kws

        print("AudioMAE model initialized")

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, waveform, device):
        # overrride the methods of the model to encode without file path
        self.model.encoder.eval()

        melspec = self.waveform_to_melspec(
            waveform
        )  # (length, n_freq_bins) = (1024, 128)
        melspec = melspec[
            :, None, :, :
        ]  # (1, 1, length, n_freq_bins) = (1, 1, 1024, 128)
        z = self.model.encoder.forward_features(
            melspec.to(device)
        )  # (b, 1+n, d); d=768
        z = z[:, 1:, :]  # (b n d); remove [CLS], the class token

        b, c, w, h = melspec.shape  # w: temporal dim; h:freq dim
        wprime = round(
            w / self.model.encoder.patch_embed.patch_size[0]
        )  # width in the latent space
        hprime = round(
            h / self.model.encoder.patch_embed.patch_size[1]
        )  # height in the latent space

        # reconstruct the temporal and freq dims
        z = rearrange(z, "b (w h) d -> b d h w", h=hprime)  # (b d h' w')

        return z

    @torch.no_grad()
    def extract_features(self, x):
        device = next(self.model.encoder.parameters()).device
        out_ = self.encode(x, device)
        return {"latents": out_}

    def waveform_to_melspec(self, waveform: torch.FloatTensor):
        # Compute the Mel spectrogram using Kaldi-compatible features
        # the parameters are chosen as described in the audioMAE paper (4.2 implementation details)

        b, c, w = waveform.shape

        mel_spectrogram = []
        for i in range(b):
            mel_spectrogram_ = kaldi.fbank(
                waveform[i, ...],
                num_mel_bins=128,
                frame_length=25.0,
                frame_shift=10.0,
                htk_compat=True,
                use_energy=False,
                sample_frequency=16000,
                window_type="hanning",
                dither=0.0,
            )
            expected_frames = 1024  # as described in the paper
            current_frames = mel_spectrogram_.shape[0]
            if current_frames > expected_frames:
                mel_spectrogram_ = mel_spectrogram_[:expected_frames, :]
            elif current_frames < expected_frames:
                padding = expected_frames - current_frames
                mel_spectrogram_ = torch.nn.functional.pad(
                    mel_spectrogram_,
                    (
                        0,
                        0,  # (left, right) for the 1st dim
                        0,
                        padding,
                    ),  # (left, right) for the 2nd dim
                )

            # scale
            # as in the AudioMAE implementation [REF: https://github.com/facebookresearch/AudioMAE/blob/bd60e29651285f80d32a6405082835ad26e6f19f/dataset.py#L300]
            mel_spectrogram_ = (mel_spectrogram_ - self.model.encoder.MEAN) / (
                self.model.encoder.STD * 2
            )  # (length, n_freq_bins) = (1024, 128)
            mel_spectrogram.append(mel_spectrogram_)

        mel_spectrogram = torch.stack(mel_spectrogram, dim=0)

        # Ensure the output shape matches 1x1024x128 by padding or trimming the time dimension
        return mel_spectrogram

    def forward(self, x, pooled=True):
        # override kws with self.extract_kws if they are changed
        if self.extract_kws:
            pooled = self.extract_kws.get("pooled", pooled)

        feature = self.extract_features(x)["latents"]
        if pooled:
            feature = feature.mean(dim=-1).mean(dim=-1)  # squish latent x and y dims

        return feature
