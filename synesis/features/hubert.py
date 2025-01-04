import torch
from torch import nn
from transformers import AutoModel, AutoProcessor


class HuBERT(nn.Module):
    def __init__(self, feature_extractor=True):
        super(HuBERT, self).__init__()

        self.feature_extractor = feature_extractor

        self.processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
        self.model = AutoModel.from_pretrained("facebook/hubert-base-ls960")

    def forward(self, x):
        if self.feature_extractor:
            with torch.no_grad():
                inputs = self.processor(x)
                h = self.model(**inputs).last_hidden_state.mean(dim=1)
                return h
        else:
            raise NotImplementedError("Training not implemented yet.")
