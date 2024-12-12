from torch import nn
from torch.optim import Adam

predict_transform_configs = {
    "pitch_shift": {
        "model": {
            "type": "regressor",
            "params": {"hidden_units": [512, 256]},
        },
        "training": {
            "criterion": nn.MSELoss,
            "optimizer": {"class": Adam, "params": {"lr": 0.001, "weight_decay": 0.01}},
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 10,
            "feature_aggregation": False,
        },
        "evaluation": {
            "criterion": nn.MSELoss,
            "feature_aggregation": False,
            "batch_size": 32,
            "metrics": [
                {
                    "name": "MSE",
                    "class": nn.MSELoss,
                    "params": {},
                },
            ],
        },
    },
    "time_stretch": {
        "model": {
            "type": "regressor",
            "params": {"hidden_units": [512, 256]},
        },
        "training": {
            "criterion": nn.MSELoss,
            "optimizer": {"class": Adam, "params": {"lr": 0.001, "weight_decay": 0.01}},
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 10,
            "feature_aggregation": False,
        },
        "evaluation": {
            "criterion": nn.MSELoss,
            "feature_aggregation": False,
            "batch_size": 32,
            "metrics": [
                {
                    "name": "MSE",
                    "class": nn.MSELoss,
                    "params": {},
                },
            ],
        },
    },
}
