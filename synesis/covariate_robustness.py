"""Methods for evaluating representation robustness to
covariate shift."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config.tasks import task_configs
from config.transforms import transform_configs
from synesis.datasets.dataset_utils import get_dataset
from synesis.downstream import train as downstream_train
from synesis.features.feature_utils import (
    DynamicBatchSampler,
    collate_packed_batch,
    get_feature_extractor,
)
from synesis.metrics import instantiate_metrics
from synesis.probes import get_probe
from synesis.transforms.transform_utils import get_transform
from synesis.utils import deep_update


def train(
    feature: str,
    dataset: str,
    task: str,
    item_format: str = "feature",
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
):
    """Train a downstream model, or load if it already exists.

    Args:
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        task: Name of the downstream task (needs to be supported by dataset).
        item_format: Format of the input data: ["audio", "feature"].
                     Defaults to "feature". If audio, feature is
                     extracted on-the-fly.
        task_config: Override certain values of the task configuration.
        device: Device to use for training (defaults to "cuda" if available).
    """
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if task_config:
        task_configs[task] = deep_update(task_configs[task], task_config)

    # check if model already exists
    model_path = Path("ckpt") / "downstream" / f"{feature}_{dataset}_{task}.pt"
    if model_path.exists():
        print(f"Loading existing downstream model from {model_path}")
        probe_cfg = task_configs[task]["model"]
        model = get_probe(
            model_type=probe_cfg["type"],
            in_features=probe_cfg["params"]["in_features"],
            n_outputs=probe_cfg["params"]["n_outputs"],
            **probe_cfg["params"],
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    else:
        model = downstream_train(
            feature=feature,
            dataset=dataset,
            task=task,
            item_format=item_format,
            task_config=task_config,
            device=device,
        )

    return model


def evaluate_representation_distance(
    feature: str,
    dataset: str,
    transform: str,
    transform_config: Optional[dict] = None,
    metric: str = "cosine",
    item_format: str = "feature",
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
):
    """
    Evaluate how much representations change when
    The input is tranformed by varying degrees.

    Args:
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        item_format: Format of the input data: ["audio", "feature"].
                Defaults to "feature". If audio, feature is
                extracted on-the-fly.
        task: Name of the downstream task.
        task_config: Override certain values of the task configuration.
        transform: Name of the transform (factor of variation).
        transform_config: Override certain values of the transform configuration.
        device: Device to use for evaluation (defaults to "cuda" if available).
        batch_size: Batch size for evaluation.
    """

    if transform_config:
        transform_configs[transform] = deep_update(
            transform_configs[transform], transform_config
        )

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    clean_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="test",
        download=False,
        item_format="feature",
    )

    transform_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="test",
        download=False,
        item_format="audio",
    )

    assert (
        transform in clean_dataset.transforms
    ), f"Transform {transform} not available in {dataset}"

    clean_sampler = DynamicBatchSampler(dataset=clean_dataset, batch_size=batch_size)
    clean_loader = DataLoader(
        clean_dataset,
        batch_sampler=clean_sampler,
        collate_fn=collate_packed_batch,
    )

    transform_sampler = DynamicBatchSampler(
        dataset=transform_dataset, batch_size=batch_size
    )
    transform_loader = DataLoader(
        transform_dataset,
        batch_sampler=transform_sampler,
        collate_fn=collate_packed_batch,
    )

    feature_extractor = get_feature_extractor(feature).to(device)

    # We will iterate over all degrees of the transform, computing distances
    # for all representations in the dataset for each.

    # for each transform, there's a param starting from "min" and one from "max"
    # that we need to find in order to define the first and last transform
    min_key = ""
    max_key = ""
    transform_params = transform_configs[transform]["params"]
    for key in transform_params:
        if key.startswith("min"):
            min_key = key
            max_key = key.replace("min", "max")
            break
    if not min_key or not max_key:
        raise (ValueError("Could not find min and max keys in transform params"))

    param_values = range(
        transform_configs[transform]["params"][min_key],
        transform_configs[transform]["params"][max_key],
        transform_configs[transform]["step"],
    )

    results = {}
    # Evaluation loop
    for pv in param_values:
        # Replace parameter range with the specific value for both min and maxs
        controlled_transform_config = transform_configs[transform].copy()
        controlled_transform_config["params"][min_key] = pv
        controlled_transform_config["params"][max_key] = pv
        transform_obj = get_transform(controlled_transform_config)

        # Iterate through clean embeddings and audio, transforming the audio
        # and computing features from it.
        for (clean_rep_batch, _), (audio_batch, _) in zip(
            clean_loader, transform_loader
        ):
            audio_batch = audio_batch.to(device)
            clean_rep_batch = clean_rep_batch.to(device)

            transformed_audio, _ = transform_obj(audio_batch)
            transformed_rep_batch = feature_extractor(transformed_audio)

            # Compute distance between clean and transformed representations
            if metric == "cosine":
                dist = 1 - torch.nn.functional.cosine_similarity(
                    clean_rep_batch, transformed_rep_batch
                )
            elif metric == "euclidean":
                dist = torch.nn.functional.pairwise_distance(
                    clean_rep_batch, transformed_rep_batch
                )
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if pv not in results:
                results[pv] = []
            results[pv].append(dist.mean().item())

    # average distances for each pv
    for pv in results:
        results[pv] = sum(results[pv]) / len(results[pv])

    return results


def evaluate_model_predictions(
    model: nn.Module,
    feature: str,
    dataset: str,
    transform: str,
    task: str,
    item_format: str = "audio",
    transform_config: Optional[dict] = None,
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
):
    """
    Evaluate downstream model predictions when the input is
    transformed by varying degrees.

    Args:
        model: Downstream model.
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        item_format: Format of the input data: ["audio", "feature"].
                Defaults to "audio".
        task: Name of the downstream task.
        task_config: Override certain values of the task configuration.
        transform: Name of the transform (factor of variation).
        transform_config: Override certain values of the transform configuration.
        device: Device to use for evaluation (defaults to "cuda" if available).
        batch_size: Batch size for evaluation.
    """

    if transform_config:
        transform_configs[transform] = deep_update(
            transform_configs[transform], transform_config
        )

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="test",
        download=False,
        item_format="audio",
    )

    assert (
        transform in test_dataset.transforms
    ), f"Transform {transform} not available in {dataset}"

    test_sampler = DynamicBatchSampler(dataset=test_dataset, batch_size=batch_size)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_packed_batch,
    )

    metrics = instantiate_metrics(
        metric_configs=task_configs[task]["evaluation"]["metrics"],
        num_classes=len(test_dataset[0][1]),
    )

    # We will iterate over all degrees of the transform, computing distances
    # for all representations in the dataset for each.

    # for each transform, there's a param starting from "min" and one from "max"
    # that we need to find in order to define the first and last transform
    min_key = ""
    max_key = ""
    transform_params = transform_configs[transform]["params"]
    for key in transform_params:
        if key.startswith("min"):
            min_key = key
            max_key = key.replace("min", "max")
            break
    if not min_key or not max_key:
        raise (ValueError("Could not find min and max keys in transform params"))

    param_values = range(
        transform_configs[transform]["params"][min_key],
        transform_configs[transform]["params"][max_key],
        transform_configs[transform]["step"],
    )

    # Evaluation loop
    model.to(device)
    model.eval()
    results = {}

    for pv in param_values:
        # Replace parameter range with the specific value for both min and maxs
        controlled_transform_config = transform_configs[transform].copy()
        controlled_transform_config["params"][min_key] = pv
        controlled_transform_config["params"][max_key] = pv

        # Iterate through clean embeddings and audio, transforming the audio
        # and computing features from it.
        total_loss = 0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for audio_batch, targets in test_loader:
                audio_batch = audio_batch.to(device)

                transform_obj = get_transform(controlled_transform_config)
                transformed_audio, _ = transform_obj(audio_batch)

                with torch.no_grad():
                    output = model(transformed_audio)
                    total_loss += model.loss(output, targets).item()

                # Store outputs and targets for metric calculation
                test_outputs.append(output)
                test_targets.append(targets)

        # Concatenate all outputs and targets
        test_outputs = torch.cat(test_outputs, dim=0)
        test_targets = torch.cat(test_targets, dim=0)

        # Calculate metrics
        results[pv] = {}
        for metric_cfg, metric in zip(
            task_configs[task]["evaluation"]["metrics"], metrics
        ):
            results[pv][metric_cfg["name"]] = metric(test_outputs, test_targets).item()

        avg_loss = total_loss / len(test_loader)
        print(f"Avg test loss: {avg_loss:.4f}")

        for name, value in results.items():
            print(f"{name}: {value:.4f}")

    return results


def evaluate_prediction_uncertainty(
    model: nn.Module,
    feature: str,
    dataset: str,
    transform: str,
    task: str,
    uncertainty_metric: str = "entropy",
    item_format: str = "audio",
    transform_config: Optional[dict] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
):
    """
    Evaluate model prediction uncertainty when the input is transformed.

    Args:
        model: Downstream model
        uncertainty_metric: Method to compute uncertainty ["entropy", "max_prob"]
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        item_format: Format of the input data: ["audio", "feature"].
                Defaults to "audio".
        task: Name of the downstream task.
        task_config: Override certain values of the task configuration.
        transform: Name of the transform (factor of variation).
        transform_config: Override certain values of the transform configuration.
        device: Device to use for evaluation (defaults to "cuda" if available).
        batch_size: Batch size for evaluation.
    """
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup dataset and loader similar to evaluate_model_predictions
    test_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="test",
        download=False,
        item_format="audio",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_packed_batch,
    )

    # We will iterate over all degrees of the transform, computing distances
    # for all representations in the dataset for each.

    # for each transform, there's a param starting from "min" and one from "max"
    # that we need to find in order to define the first and last transform
    min_key = ""
    max_key = ""
    transform_params = transform_configs[transform]["params"]
    for key in transform_params:
        if key.startswith("min"):
            min_key = key
            max_key = key.replace("min", "max")
            break
    if not min_key or not max_key:
        raise (ValueError("Could not find min and max keys in transform params"))

    param_values = range(
        transform_configs[transform]["params"][min_key],
        transform_configs[transform]["params"][max_key],
        transform_configs[transform]["step"],
    )

    results = {}
    model.eval()

    for pv in param_values:
        # Replace parameter range with the specific value for both min and maxs
        controlled_transform_config = transform_configs[transform].copy()
        controlled_transform_config["params"][min_key] = pv
        controlled_transform_config["params"][max_key] = pv
        transform_obj = get_transform(controlled_transform_config)

        uncertainties = []

        with torch.no_grad():
            for audio_batch, _ in test_loader:
                audio_batch = audio_batch.to(device)

                # Apply transform
                transformed_audio, _ = transform_obj(audio_batch)

                # Get model predictions
                logits = model(transformed_audio)
                probs = torch.softmax(logits, dim=1)

                # Compute uncertainty
                if uncertainty_metric == "entropy":
                    uncertainty = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                elif uncertainty_metric == "max_prob":
                    uncertainty = 1 - probs.max(dim=1)[0]
                else:
                    raise ValueError(
                        f"Unknown uncertainty metric: {uncertainty_metric}"
                    )

                uncertainties.extend(uncertainty.cpu().numpy())

        results[pv] = {
            "mean": float(np.mean(uncertainties)),
            "std": float(np.std(uncertainties)),
        }

    return results
