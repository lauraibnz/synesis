"""Methods for evaluating how disentangled two factors of variation are.
It does this by measuring the evaluating a model trained to predict
one factor of variation on input data that is subjected to a transformation
corresponding to the other factor of variation, and vice versa."""

import argparse
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
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
from synesis.utils import deep_update, get_artifact, get_metric_from_wandb, get_wandb_config


def preprocess_batch(
    batch_raw_data, transform_obj, transform, sample_rate, feature_extractor, device
):
    if any(
        tf in transform
        for tf in ["HueShift", "BrightnessShift", "SaturationShift", "JPEGCompression", "InstrumentShift"]
    ):
        transformed_raw_data = batch_raw_data[:, 1].to(device)
    elif "TimeStretch" in transform:
        original_np = batch_raw_data.to("cpu").numpy()
        transformed_list = []
        for i in range(original_np.shape[0]):
            original_len = original_np.shape[2]
            np_item = transform_obj(original_np[i][0], sample_rate=sample_rate)
            if len(np_item) > original_len:
                # Random crop to original length
                offset = torch.randint(0, len(np_item) - original_len, (1,)).item()
                np_item = np_item[offset : offset + original_len]
            elif len(np_item) < original_len:
                # Repeat-pad to original length, split randomly left/right
                pad = original_len - len(np_item)
                left_pad = torch.randint(0, pad + 1, (1,)).item()
                right_pad = pad - left_pad
                # Wrap pad using numpy for speed and consistency
                np_item = np.pad(np_item, (left_pad, right_pad), mode="wrap")
            # Convert to tensor
            transformed_list.append(torch.as_tensor(np_item, dtype=torch.float32))
        transformed_raw_data = torch.stack(transformed_list, dim=0).to(device)
        if transformed_raw_data.dim() == 2:
            transformed_raw_data = transformed_raw_data.unsqueeze(1)
    else:
        transformed_raw_data = transform_obj(batch_raw_data.to(device))

    transformed_features = feature_extractor(transformed_raw_data)
    transformed_features = transformed_features.to(device)
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
    batch_size: int = 32,
    logging: bool = False,
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

    feature_config = feature_configs.get(feature)
    transform_config = transform_configs.get(transform)
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )

    if logging:
        wandb_config = get_wandb_config()
        run_name = f"DISENT_{transform}_{dataset}_{feature}_{label}"
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=run_name,
            config={
                "feature": feature,
                "feature_config": feature_config,
                "dataset": dataset,
                "task": task,
                "task_config": task_config,
            },
        )
        artifact = wandb.Artifact(run_name, type="model")

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
        itemization=False,
    )

    # If dataset returns subitems per item, need to wrap it
    if transform_config and raw_dataset[0][0].dim() == 3:
        wrapped_dataset = SubitemDataset(raw_dataset)
        del raw_dataset
        raw_dataset = wrapped_dataset

    dataloader = DataLoader(
        raw_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    feature_extractor = get_feature_extractor(feature)
    feature_extractor = feature_extractor.to(device)

    sample_item, _ = raw_dataset[0]

    if any(
        tf in transform
        for tf in ["HueShift", "BrightnessShift", "SaturationShift", "JPEGCompression", "InstrumentShift"]
    ):
        sample_item = sample_item[0]

    with torch.no_grad():
        extracted_features = feature_extractor(sample_item)

    if extracted_features.dim() == 1:
        in_features = extracted_features.shape[0]
    else:
        in_features = extracted_features.shape[1]

    if extracted_features.dim() == 3:
        use_temporal_pooling = True
    else:
        use_temporal_pooling = False

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
            in_features=in_features,
            n_outputs=n_outputs,
            use_temporal_pooling=use_temporal_pooling,
            **task_config["model"]["params"],
        ).to(device)
        model.load_state_dict(torch.load(Path(artifact_dir) / f"{feature}.pt"))
        os.remove(Path(artifact_dir) / f"{feature}.pt")

    model.to(device)
    model.eval()

    if any(tf in transform for tf in ["PitchShift", "AddWhiteNoise"]):
        transform_obj = get_transform(
            transform_config,
            sample_rate=feature_config["sample_rate"],
        )
    elif not transform_config:
        # transform handled in dataset
        transform_obj = None
    else:
        transform_obj = get_transform(transform_config)

    total_loss = 0
    criterion = task_config["evaluation"]["criterion"]()

    # Instantiate metrics from the task configuration
    from synesis.metrics import instantiate_metrics
    metrics = instantiate_metrics(
        metric_configs=task_config["evaluation"]["metrics"],
        num_classes=n_outputs,
    )

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
            if len(predictions.shape) > 1 and n_outputs == 1:
                predictions = predictions.squeeze(1)
            
            # Handle target dtype for loss computation
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                target_for_loss = batch_targets.float()
            else:
                target_for_loss = batch_targets
            
            loss = criterion(predictions, target_for_loss.to(device))
            total_loss += loss.item()

            # Handle target dtype for metrics (metrics expect long for classification)
            if task_config["model"]["type"] == "regressor":
                target_for_metrics = target_for_loss
            else:
                # For classification tasks, metrics expect long tensors
                target_for_metrics = batch_targets.long()

            # Update metrics
            for metric_cfg, metric in zip(task_config["evaluation"]["metrics"], metrics):
                metric = metric.to(device)
                metric.update(predictions, target_for_metrics.to(device))

    mean_loss = total_loss / len(dataloader)
    print(f"Mean loss: {mean_loss}")

    # Calculate and print metrics from the task configuration
    test_metric_results = {}
    for metric_cfg, metric in zip(task_config["evaluation"]["metrics"], metrics):
        test_metric_results[metric_cfg["name"]] = metric.compute().item()
        metric.reset()
        print(f"{metric_cfg['name']}: {test_metric_results[metric_cfg['name']]:.4f}")

    results = {"avg_loss": mean_loss, **test_metric_results}

    # Get original metrics from wandb run
    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    original_metrics = {name: get_metric_from_wandb(run, f"{name}") for name in test_metric_results.keys()}

    # Calculate differences
    diff_metrics = {
        f"diff_{k}": original_metrics.get(k, 0) - v for k, v in test_metric_results.items()
    }

    for name in test_metric_results.keys():
        diff_val = diff_metrics[f"diff_{name}"]
        print(f"{name} difference: {diff_val:.4f}")

    # Log only the new metrics and differences
    log_results = {
        **{f"test/{name}": value for name, value in test_metric_results.items()},
        **diff_metrics,
    }

    if logging:
        wandb.log(log_results)


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
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        required=False,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--nolog",
        action="store_true",
        help="Do not log to wandb.",
    )

    args = parser.parse_args()

    # need to get run_id from wandb
    wandb_config = get_wandb_config()
    entity = wandb_config["entity"]
    project = wandb_config["project"]
    wandb_runs = wandb.Api().runs(f"{entity}/{project}")
    run_name = f"INFO_DOWN_{args.task}_{args.dataset}_{args.label}_{args.feature}"
    run_id = None
    for run in wandb_runs:
        if run.name == run_name:
            run_id = run.id
    if run_id is None:
        raise ValueError(f"Run {run_name} not matched.")

    evaluate_disentanglement(
        model=f"{entity}/{project}/{run_id}/{run_name}",
        feature=args.feature,
        dataset=args.dataset,
        transform=args.transform,
        label=args.label,
        task=args.task,
        device=args.device,
        batch_size=args.batch_size,
        logging=not args.nolog,
    )
