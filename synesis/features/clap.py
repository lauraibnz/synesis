import os

import laion_clap
import wget
from torch import nn


class CLAP(nn.Module):
    def __init__(self, feature_extractor=True):
        super(CLAP, self).__init__()

        self.model = laion_clap.CLAP_Module(
            enable_fusion=False, amodel="HTSAT-base"
        ).to("cuda")

        if not os.path.exists("models/pretrained/music_speech_epoch_15_esc_89.25.pt"):
            print("Download pretrained model for speech-music...")
            url = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt"
            wget.download(url, "models/pretrained/music_speech_epoch_15_esc_89.25.pt")

        self.model.load_ckpt("models/pretrained/music_speech_epoch_15_esc_89.25.pt")

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        features = self.model.get_audio_embedding_from_data(x=x, use_tensor=True)
        return features
