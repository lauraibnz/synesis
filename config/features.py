configs = {
    "MDuo": {
        "__cls__": "MDuo",
        "item_len_sec": 10,
        "sample_rate": 16000,
        "feature_dim": 3840,
        "extract_kws": {
            "pooled": True,
        },
    },
    "AudioMAE": {
        "__cls__": "AudioMAE",
        "item_len_sec": 10,
        "sample_rate": 16000,
        "feature_dim": 768,
        "extract_kws": {
            "pooled": True,
        },
    },
    "VGGishMTAT": {
        "__cls__": "VGGishMTAT",
        "item_len_sec": 3.69,
        "sample_rate": 16000,
        "feature_dim": 512,
    },
    "MULE_1728": {
        "__cls__": "MULE",
        "item_len_sec": 2.99,
        "sample_rate": 16000,
        "feature_dim": 1728,
        "extract_kws": {
            "key": "encoded",
        },
    },
    "MULE_512": {
        "__cls__": "MULE",
        "item_len_sec": 2.99,
        "sample_rate": 16000,
        "feature_dim": 512,
        "extract_kws": {
            "key": "projected",
        },
    },
    "Music2Latent_8192": {
        "__cls__": "MusicLatent",
        "item_len_sec": 3,
        "sample_rate": 44100,
        "feature_dim": 8192,
        "extract_kws": {
            "extract_features": True,
        },
    },
    "Music2Latent_64": {
        "__cls__": "MusicLatent",
        "item_len_sec": 3,
        "sample_rate": 44100,
        "feature_dim": 64,
        "extract_kws": {
            "extract_features": False,
        },
    },
    "PESTO": {
        "__cls__": "PESTO",
        "item_len_sec": 3,
        "sample_rate": 22050,
        "feature_dim": 384,
    },
    "MERT": {
        "__cls__": "MERT",
        "item_len_sec": 3,
        "sample_rate": 24000,
        "feature_dim": 768,
        "extract_kws": {
            "pooled": True,
        },
    },
    "ResNet50_ImageNet": {
        "__cls__": "ResNet50_ImageNet",
        "resize_dim": 256,
        "input_dim": 224,
        "feature_dim": 2048,
    },
    "SimCLR": {
        "__cls__": "SimCLR",
        "resize_dim": 256,
        "input_dim": 224,
        "feature_dim": 2048,
    },
}
