from torch import nn
from torch.optim import Adam
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score

task_configs = {
    "tagging": {
        "model": {
            "type": "classifier",
            "params": {"hidden_units": [512, 256]},
        },
        "training": {
            "criterion": nn.BCEWithLogitsLoss,
            "optimizer": {"class": Adam, "params": {"lr": 0.001, "weight_decay": 0.01}},
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 10,
            "feature_aggregation": True,
        },
        "evaluation": {
            "feature_aggregation": True,
            "batch_size": 32,
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
    "pitch_classification": {
        "model": {
            "type": "classifier",
            "params": {"hidden_units": [512, 256]},
        },
        "training": {
            "criterion": nn.CrossEntropyLoss,
            "optimizer": {"class": Adam, "params": {"lr": 0.001, "weight_decay": 0.01}},
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 10,
            "feature_aggregation": True,
        },
        "evaluation": {
            "feature_aggregation": True,
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
    "instrument_classification": {
        "model": {
            "type": "classifier",
            "params": {"hidden_units": [512, 256]},
        },
        "training": {
            "criterion": nn.CrossEntropyLoss,
            "optimizer": {"class": Adam, "params": {"lr": 0.001, "weight_decay": 0.01}},
            "batch_size": 32,
            "num_epochs": 100,
            "patience": 10,
            "feature_aggregation": True,
        },
        "evaluation": {
            "feature_aggregation": True,
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
}
