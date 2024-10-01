from torch import nn
from torch.optim import Adam


task_config = {
    "tagging": {
        "model": {
            "type": "classifier",
            "params": {"hidden_units": [512, 256]},
        },
        "training": {
            "loss": nn.BCEWithLogitsLoss,
            "optimizer": {
                "class": Adam,
                "params": {"lr": 0.001, "weight_decay": 0.01}
            },
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 10,
            "feautre_aggregation": True
        },
        "evaluation": {
            "feature_aggregation": True,
            "batch_size": 32,
        },
    },
    "pitch_class_classification": {
        "model": {
            "type": "classifier",
            "params": {"hidden_units": [512, 256]},
        },
        "training": {
            "loss": nn.CrossEntropyLoss,
            "optimizer": {
                "class": Adam,
                "params": {"lr": 0.001, "weight_decay": 0.01}
            },
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 10,
            "feature_aggregation": True,
        },
        "evaluation": {
            "feature_aggregation": True,
            "batch_size": 32,
        },
    },
    "instrument_classification": {
        "model": {
            "type": "classifier",
            "params": {"hidden_units": [512, 256]},
        },
        "training": {
            "loss": nn.CrossEntropyLoss,
            "optimizer": {
                "class": Adam,
                "params": {"lr": 0.001, "weight_decay": 0.01}
            },
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 10,
            "feature_aggregation": True,
        },
        "evaluation": {
            "feature_aggregation": True,
            "batch_size": 32,
        },
    },
}
