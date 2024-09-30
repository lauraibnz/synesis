from torch import nn
from torch.optim import Adam


task_config = {
    "tagging": {
        "loss": nn.BCEWithLogitsLoss,
        "optimizer": {"class": Adam, "params": {"lr": 0.001, "patience": 10}},
        "model": {
            "type": "classifier",
            "params": {"hidden_units": [512, 256], "weight_decay": 0.01},
        },
    },
    "pitch_class_classification": {
        "loss": nn.CrossEntropyLoss,
        "optimizer": {"class": Adam, "params": {"lr": 0.001, "patience": 10}},
        "model": {
            "type": "classifier",
            "params": {"hidden_units": [512, 256], "weight_decay": 0.01},
        },
    },
    "instrument_classification": {
        "loss": nn.CrossEntropyLoss,
        "optimizer": {"class": Adam, "params": {"lr": 0.001, "patience": 10}},
        "model": {
            "type": "classifier",
            "params": {"hidden_units": [512, 256], "weight_decay": 0.01},
        },
    },
}
