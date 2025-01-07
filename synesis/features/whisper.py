import torch
from torch import nn
from transformers import WhisperModel, WhisperProcessor


class Whisper(nn.Module):
    def __init__(self):
        super(Whisper, self).__init__()
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperModel.from_pretrained("openai/whisper-small").to("cuda")
        self.max_batch_size = 1  # Maximum batch size for VRAM constraints

        # Constants
        self.sample_rate = 16000
        self.expected_seconds = 30
        self.clip_seconds = 5
        self.frames_per_second = 50
        self.frames_per_clip = self.clip_seconds * self.frames_per_second

    def process_batch(self, x):
        batch_size = x.shape[0]
        padded_length = self.expected_seconds * self.sample_rate
        padded_audio = torch.zeros((batch_size, padded_length), device=x.device)
        padded_audio[:, : x.shape[1]] = x

        inputs = self.processor(
            padded_audio.cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        outputs = self.model.encoder(inputs.input_features.to(self.model.device))
        features = outputs.last_hidden_state[:, : self.frames_per_clip, :]
        features = torch.mean(features, dim=1)
        return features

    def forward(self, x):
        batch_size = x.shape[0]
        feature_list = []

        # Process in smaller batches
        for i in range(0, batch_size, self.max_batch_size):
            batch = x[i : i + self.max_batch_size]
            features = self.process_batch(batch)
            feature_list.append(features)

        # Concatenate all batches
        return torch.cat(feature_list, dim=0)
