import torch
from torch import nn
from transformers import UniSpeechSatForCTC, Wav2Vec2Processor


class UniSpeech(nn.Module):
    def __init__(self, feature_extractor=True):
        super(UniSpeech, self).__init__()

        self.processor = Wav2Vec2Processor.from_pretrained(
            "microsoft/unispeech-sat-base-100h-libri-ft"
        )
        self.model = UniSpeechSatForCTC.from_pretrained(
            "microsoft/unispeech-sat-base-100h-libri-ft", output_hidden_states=True
        ).to("cuda")

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        input_values = self.processor(
            x, return_tensors="pt", padding="longest", sampling_rate=16000
        ).input_values
        if input_values.dim() == 3 and input_values.shape[0] == 1:
            input_values = input_values.squeeze(0)
        input_values = input_values.to("cuda")
        features = self.model(input_values).hidden_states[-1]

        # mean over time
        features = torch.mean(features, dim=1)

        return features
