from torch import nn
from torch.optim import Adam
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score
from synesis.utils import NoteMetrics
from synesis.losses import MaskedBce

configs = {
    "default": {
        "model": {
            "type": "classifier",
            "params": {"hidden_units": [512, 256]},
        },
        "training": {
            "criterion": nn.CrossEntropyLoss,
            "optimizer": {
                "class": Adam,
                "params": {"lr": 0.001, "weight_decay": 0.001},
            },
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 10,
            "feature_aggregation": False,
        },
        "evaluation": {
            "criterion": nn.CrossEntropyLoss,
            "feature_aggregation": False,
            "batch_size": 32,
            "metrics": [
                {
                    "name": "Accuracy",
                    "class": Accuracy,
                    "params": {
                        "task": "multiclass",
                    },
                },
                {
                    "name": "F1",
                    "class": F1Score,
                    "params": {
                        "task": "multiclass",
                    },
                },
            ],
        },
    },
    "regression_MLP": {
        "model": {
            "type": "regressor",
        },
        "training": {
            "criterion": nn.MSELoss,
        },
        "evaluation": {
            "criterion": nn.MSELoss,
            "metrics": [
                {"name": "MSE", "class": nn.MSELoss, "params": {}},
            ],
        },
    },
    "regression_SLP": {
        "model": {
            "type": "regressor",
            "params": {"hidden_units": []},
        },
        "training": {
            "criterion": nn.MSELoss,
            "feature_aggregation": False,
        },
        "evaluation": {
            "criterion": nn.MSELoss,
            "metrics": [
                {"name": "MSE", "class": nn.MSELoss, "params": {}},
            ],
        },
    },
    "tagging": {
        "model": {
            "params": {"hidden_units": []},
        },
        "training": {
            "criterion": nn.BCEWithLogitsLoss,
            "feature_aggregation": True,
        },
        "evaluation": {
            "metrics": [
                {"name": "AUC_ROC", "class": AUROC, "params": {"task": "multilabel"}},
                {
                    "name": "AP",
                    "class": AveragePrecision,
                    "params": {"task": "multilabel"},
                },
            ],
        },
    },
    "transcriber_probe": {
        "model": {
            "type": "transcriber",
            "params": {"hidden_units": [512]},
        },
        "training": {
            "criterion": MaskedBce,
            "optimizer": {
                "class": Adam,
                "params": {"lr": 0.00005, "weight_decay": 0.00001},
            },
            "batch_size": 2,
            "num_epochs": 20,
            "feature_aggregation": True,
        },
        "evaluation": {
            "criterion": MaskedBce,
            "feature_aggregation": True,
            "batch_size": 2,
            "metrics": [
                {"name": "F1", "class": F1Score,"params": {"task": "binary",},},
                {"name": "NoteMetrics", "class": NoteMetrics, "params": {"hop_secs": 1/62.5}},
            ],
        },
    },
}
