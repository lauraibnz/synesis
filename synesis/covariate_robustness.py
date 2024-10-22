"""Methods for evaluating representation robustness to
covariate shift."""


from pathlib import Path
import torch
from synesis.downstream import train as downstream_train
from synesis.probes import get_probe
from config.tasks import task_config as tc


def train(feature, dataset, task, device=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # check if model already exists
    model_path = Path("ckpt") / "downstream" / f"{feature}_{dataset}_{task}.pt"
    if model_path.exists():
        print(f"Loading existing downstream model from {model_path}")
        probe_cfg = tc[task]["model"]
        model = get_probe(
            model_type=probe_cfg['type'],
            in_features=probe_cfg['params']['in_features'],
            n_outputs=probe_cfg['params']['n_outputs'],
            **probe_cfg['params']
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    else:
        model = downstream_train(
            feature=feature,
            dataset=dataset,
            task=task,
            device=device
        )
