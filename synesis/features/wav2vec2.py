import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2Vec2(nn.Module):
    def __init__(self, feature_extractor=True):
        super(Wav2Vec2, self).__init__()

        self.feature_extractor = feature_extractor

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def forward(self, x):
        if self.feature_extractor:
            with torch.no_grad():
                inputs = self.processor(
                    x, return_tensors="pt", padding="longest"
                ).input_values.squeeze(0)
                h = self.model(inputs).last_hidden_state.mean(dim=1)
                return h
        else:
            raise NotImplementedError("Training not implemented yet.")
