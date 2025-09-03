from torch import nn
from torch.optim import Adam
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, MeanSquaredError, MeanAbsoluteError
from synesis.utils import NoteMetrics, F1Metrics, AccMetrics
from synesis.losses import MaskedBce

TRANSCRIBER_FRAME_RATE = 62.75

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
                {"name": "MSE", "class": MeanSquaredError, "params": {}},
                {"name": "MAE", "class": MeanAbsoluteError, "params": {}},
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
                {"name": "MSE", "class": MeanSquaredError, "params": {}},
                {"name": "MAE", "class": MeanAbsoluteError, "params": {}},
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
            "criterion": nn.BCEWithLogitsLoss,
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
            "params": {"hidden_units": [512, 512]},
        },
        "training": {
            "criterion": MaskedBce,
            "optimizer": {
                "class": Adam,
                "params": {"lr": 0.00005, "weight_decay": 0.00001},
            },
            "batch_size": 4,
            "num_epochs": 20,
            "feature_aggregation": True,
        },
        "evaluation": {
            "criterion": MaskedBce,
            "feature_aggregation": True,
            "batch_size": 1,
            "metrics": [
                {"name": "F1Metrics", "class": F1Metrics, "params": {"frame_rate": TRANSCRIBER_FRAME_RATE}},
                {"name": "NoteMetrics", "class": NoteMetrics, "params": {"hop_secs": 1./TRANSCRIBER_FRAME_RATE}},
                {"name": "AccMetrics", "class": AccMetrics, "params": {"frame_rate": TRANSCRIBER_FRAME_RATE}},
            ],
        },
    },
    "classification_SLP": {
        "model": {
            "params": {"hidden_units": []},
        },
    },
    "classification_MLP": {},
    "tagging_MLP": {
        "training": {
            "criterion": nn.BCEWithLogitsLoss,
            "feature_aggregation": True,
        },
        "evaluation": {
            "criterion": nn.BCEWithLogitsLoss,
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
}
