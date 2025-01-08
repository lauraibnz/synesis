from speechbrain.pretrained import EncoderClassifier


class XVector:
    def __init__(self, feature_extractor=True):
        self.encoder = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-xvect-voxceleb"
        )

    def embed(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        features = self.encoder.encode_batch(x)
        if features.dim() == 3:
            features = features.squeeze(1)

        return features


if __name__ == "__main__":
    import torch

    xvector = XVector()
    x = torch.randn(4, 16000 * 10)
    features = xvector.embed(x)
    print(features.shape)
