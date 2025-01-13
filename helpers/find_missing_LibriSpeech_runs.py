from itertools import product

import wandb


def get_equi_run_names():
    return [
        f"{eval_type}_{task}_{transform}_{label}_{ds}_{feature}"
        for eval_type, task, transform, label, ds, feature in product(
            equi_types, equi_tasks, transforms, equi_labels, dataset, features
        )
    ]


def get_info_run_names():
    return [
        f"{eval_type}_{task}_{ds}_{label}_{feature}"
        for eval_type, task, ds, label, feature in product(
            info_types, info_tasks, dataset, info_labels, features
        )
    ]


entity = "cplachouras"
project = "synesis"

dataset = ["LibriSpeech"]
equi_types = ["EQUI_PARA", "EQUI_FEAT"]
info_types = ["INFO_DOWN"]
features = [
    "MDuo",
    "AudioMAE",
    "Wav2Vec2",
    "HuBERT",
    "CLAP",
    "Whisper",
    "UniSpeech",
    "XVector",
]
info_tasks = ["regression", "regression_linear"]
equi_tasks = ["regression", "regression_linear"]
equi_labels = ["dummy"]
info_labels = ["wps"]
transforms = ["PitchShift", "AddWhiteNoise", "TimeStretch"]

wandb_runs = wandb.Api().runs(f"{entity}/{project}")
run_names = get_equi_run_names() + get_info_run_names()
wandb_runs = [str(run.name) for run in wandb_runs if "LibriSpeech" in run.name]

# print differences between run_names and wandb_runs
print("Not in wandb:")
for run_name in run_names:
    if run_name not in wandb_runs:
        print(run_name)
print("Not in run_names:")
for wandb_run in wandb_runs:
    if wandb_run not in run_names:
        print(wandb_run)
