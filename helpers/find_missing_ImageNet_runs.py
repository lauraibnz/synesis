from collections import Counter
from itertools import product

import wandb


def get_equi_run_names():
    return [
        f"{eval_type}_{task}_{transform_map[label]}_{label}_{ds}_{feature}"
        for eval_type, task, label, ds, feature in product(
            equi_types, tasks, labels, dataset, features
        )
    ]


def get_info_run_names():
    return [
        f"{eval_type}_{task}_{ds}_{label}_{feature}"
        for eval_type, task, ds, label, feature in product(
            info_types, tasks, dataset, labels, features
        )
    ]


entity = "cplachouras"
project = "synesis"

dataset = ["ImageNet"]
equi_types = ["EQUI_PARA", "EQUI_FEAT"]
info_types = ["INFO_DOWN"]
features = [
    "ResNet50_ImageNet",
    "ViT_ImageNet",
    "SimCLR",
    "DINO",
    "ViT_MAE",
    "CLIP",
    "IJEPA",
]
tasks = ["regression", "regression_linear"]
labels = ["hue", "saturation", "brightness"]
transform_map = {
    "hue": "HueShift",
    "saturation": "SaturationShift",
    "brightness": "BrightnessShift",
}

wandb_runs = wandb.Api().runs(f"{entity}/{project}")
run_names = get_equi_run_names() + get_info_run_names()
wandb_runs = [str(run.name) for run in wandb_runs if "ImageNet" in run.name]

# print differences between run_names and wandb_runs
print("Not in wandb:")
for run_name in run_names:
    if run_name not in wandb_runs:
        print(run_name)
print("Not in run_names:")
for wandb_run in wandb_runs:
    if wandb_run not in run_names:
        print(wandb_run)

# check if there are duplicates in wandbruns
print("Duplicates:")
print([k for k, v in Counter(wandb_runs).items() if v > 1])
