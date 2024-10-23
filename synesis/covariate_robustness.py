"""Methods for evaluating representation robustness to
covariate shift."""

from pathlib import Path
from typing import Optional

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
    get_pretrained_model,
)
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
    model: nn.Module,
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
        model: Trained downstream model.
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

    feature_extractor = get_pretrained_model(feature).to(device)

    model.eval()

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
