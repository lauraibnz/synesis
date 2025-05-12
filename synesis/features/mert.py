"""MERT.
https://github.com/yizhilll/MERT
License: Apache License 2.0 (see NOTICE for full license)
"""

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn


class MERT(nn.Module):
    def __init__(self, feature_extractor=False, extract_kws={}, **kwargs):
        super(MERT, self).__init__()

        self.model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        self.feature_extractor = feature_extractor  # different than extract_features, which is generation vs analysis
        self.extract_kws = extract_kws

        print("MERT model initialized")

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_features(self, x):
        device = next(self.model.parameters()).device

        inputs = self.processor(x, sampling_rate=24000, return_tensors="pt")
        inputs["input_values"] = inputs["input_values"].squeeze(0).to(device)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0).to(device)

        out_ = self.model(**inputs, output_hidden_states=True)["last_hidden_state"]
        return {"latents": out_}

    def forward(self, x, pooled=True):
        if x.dim() == 3:  # stereo
            x = x.mean(dim=1)

        # override kws with self.extract_kws if they are changed
        if self.extract_kws:
            pooled = self.extract_kws.get("pooled", pooled)

        feature = self.extract_features(x)
        if pooled:
            feature = feature["latents"].mean(dim=-2)

        return feature
