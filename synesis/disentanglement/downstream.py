"""Methods for evaluating how disentangled two factors of variation are.
It does this by measuring the evaluating a model trained to predict
one factor of variation on input data that is subjected to a transformation
corresponding to the other factor of variation, and vice versa."""

import argparse
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config.features import configs as feature_configs
from config.informativeness.downstream import configs as task_configs
from config.transforms import configs as transform_configs
from synesis.datasets.dataset_utils import SubitemDataset, get_dataset
from synesis.features.feature_utils import get_feature_extractor
from synesis.probes import get_probe
from synesis.transforms.transform_utils import get_transform
from synesis.utils import deep_update, get_artifact


def preprocess_batch(
    batch_raw_data, transform_obj, transform, sample_rate, feature_extractor, device
):
    if any(
        tf in transform
        for tf in ["HueShift", "BrightnessShift", "SaturationShift", "JPEGCompression"]
    ):
        transformed_raw_data = batch_raw_data[:, 1].to(device)
    elif "TimeStretch" in transform:
        raise (NotImplementedError("TimeStretch not implemented yet."))
    else:
        transformed_raw_data = transform_obj(batch_raw_data.to(device))

    transformed_features = feature_extractor(transformed_raw_data)
    return transformed_features


def evaluate_disentanglement(
    model: Union[nn.Module, str],
    feature: str,
    dataset: str,
    transform: str,
    label: str,
    task: str = "default",
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
):
    """Evaluate disentanglement between two factors of variation.
    Args:
        model: Model to evaluate.
        feature: Name of the feature to evaluate disentanglement for.
        dataset: Name of the dataset to use.
        transform: Name of the transform to apply.
        label: Name of the factor of variation to evaluate.
        task_config: Configuration for the task.
        device: Device to use for the computation.
    Returns:
        A dictionary containing mean metrics.
    """
    if dataset not in ["ImageNet", "LibriSpeech"]:
        raise ValueError(f"Invalid dataset: {dataset}")

    feature_config = feature_configs.get(feature)
    transform_config = transform_configs.get(transform)
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the dataset
    raw_dataset = get_dataset(
        name=dataset,
        feature=feature,
        transform=transform,
        label=label,
        split="test",
        download=False,
        item_format="raw",
    )

    if isinstance(model, str):
        # Load model from wandb artifact
        artifact = get_artifact(model)
        artifact_dir = artifact.download()
        n_outputs = (
            1
            if task_config["model"]["type"] == "regressor"
            else len(raw_dataset.label_encoder.classes_)
        )
        model = get_probe(
            model_type=task_config["model"]["type"],
            in_features=feature_config["feature_dim"],
            n_outputs=n_outputs,
            **task_config["model"]["params"],
        ).to(device)
        model.load_state_dict(torch.load(Path(artifact_dir) / f"{feature}.pt"))
        os.remove(Path(artifact_dir) / f"{feature}.pt")

    model.to(device)
    model.eval()

    # If dataset returns subitems per item, need to wrap it
    if dataset != "ImageNet" and raw_dataset[0][0].dim() == 3:
        wrapped_dataset = SubitemDataset(raw_dataset)
        del raw_dataset
        raw_dataset = wrapped_dataset

    if task_config["training"].get("feature_aggregation") or task_config[
        "evaluation"
    ].get("feature_aggregation"):
        raise NotImplementedError(
            "Feature aggregation is not currently implemented for transform prediction."
        )

    dataloader = DataLoader(
        raw_dataset,
        batch_size=task_config["evaluation"]["batch_size"],
        shuffle=False,
    )

    feature_extractor = get_feature_extractor(feature)
    feature_extractor = feature_extractor.to(device)

    if any(tf in transform for tf in ["PitchShift", "AddWhiteNoise"]):
        transform_obj = get_transform(
            transform_config,
            sample_rate=feature_config["sample_rate"],
        )
    elif dataset == "ImageNet":
        # transform handled in dataset
        transform_obj = None
    else:
        transform_obj = get_transform(transform_config)

    total_loss = 0
    all_predictions = []
    all_targets = []
    criterion = task_config["evaluation"]["criterion"]()

    with torch.no_grad():
        for batch_raw_data, batch_targets in tqdm(dataloader, desc="Evaluating"):
            transformed_features = preprocess_batch(
                batch_raw_data=batch_raw_data,
                transform_obj=transform_obj,
                transform=transform,
                sample_rate=feature_config.get("sample_rate", None),
                feature_extractor=feature_extractor,
                device=device,
            )

            predictions = model(transformed_features)
            if len(predictions.shape) == 2:
                predictions = predictions.squeeze(1)
            loss = criterion(predictions, batch_targets.to(device))
            total_loss += loss.item()

            all_predictions.append(predictions.detach().cpu())
            all_targets.append(batch_targets.detach().cpu())

    mean_loss = total_loss / len(dataloader)
    print(f"Mean loss: {mean_loss}")

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MSE
    mse = nn.MSELoss()(all_predictions, all_targets).item()
    print(f"MSE: {mse}")

    results = {"avg_loss": mean_loss, "mse": mse}

    # Get original metrics from wandb run
    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    original_metrics = run.summary.get("evaluation_metrics", {})

    # Calculate differences
    diff_metrics = {
        f"diff_{k}": original_metrics.get(k, 0) - v for k, v in results.items()
    }

    # Combine all results
    all_results = {
        "run_name": run_name,
        "dataset": args.dataset,
        "transform": args.transform,
        "label": args.label,
        "original_mse": original_metrics.get("mse", 0),
        "transformed_mse": results["mse"],
        "diff_mse": diff_metrics["diff_mse"],
    }

    # Save/append to CSV
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "disentanglement_results.csv"

    df = pd.DataFrame([all_results])
    if not results_file.exists():
        df.to_csv(results_file, index=False)
    else:
        df.to_csv(results_file, mode="a", header=False, index=False)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model when input is transformed."
    )
    parser.add_argument(
        "--feature",
        "-f",
        type=str,
        required=True,
        help="Feature name.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset name.",
    )
    parser.add_argument(
        "--transform",
        "-tf",
        type=str,
        required=True,
        help="Data transform name.",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=False,
        default="default",
        help="Task name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device to use for training.",
    )
    parser.add_argument(
        "--label",
        "-l",
        type=str,
        required=True,
        help="Factor of variation.",
    )

    args = parser.parse_args()

    # need to get run_id from wandb
    entity = "cplachouras"
    project = "synesis"
    wandb_runs = wandb.Api().runs(f"{entity}/{project}")
    run_name = f"INFO_DOWN_{args.task}_{args.dataset}_{args.label}_{args.feature}"
    run_id = None
    for run in wandb_runs:
        if run.name == run_name:
            run_id = run.id
            break
    if run_id is None:
        raise ValueError(f"Run {run_name} not matched.")

    results = evaluate_disentanglement(
        model=f"{entity}/{project}/{run_id}/{run_name}",
        feature=args.feature,
        dataset=args.dataset,
        transform=args.transform,
        label=args.label,
        task=args.task,
        device=args.device,
    )
