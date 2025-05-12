from torch import nn
from torch.optim import Adam

configs = {
    "default": {
        "model": {
            "type": "regressor",
            "params": {"hidden_units": [512, 512]},
            "batch_norm": False,
            "feature_norm": True,
            "emb_param": True,
            "emb_param_dim": 32,
        },
        "training": {
            "criterion": nn.MSELoss,
            "optimizer": {"class": Adam, "params": {"lr": 0.001, "weight_decay": 0.01}},
            "batch_size": 16,
            "num_epochs": 5,
            "patience": 3,
            "feature_aggregation": False,
        },
        "evaluation": {
            "criterion": nn.MSELoss,
            "feature_aggregation": False,
            "batch_size": 16,
        },
    },
    "regression_MLP": {},
    "regression_linear": {
        "model": {
            "params": {"hidden_units": []},
        },
    },
}
