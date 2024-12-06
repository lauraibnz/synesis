"""Methods for training and evaluating downstream models."""

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.features import feature_configs
from config.tasks import task_configs
from synesis.datasets.dataset_utils import AggregateDataset, SubitemDataset, get_dataset
from synesis.features.feature_utils import (
    DynamicBatchSampler,
    collate_packed_batch,
    get_feature_extractor,
)
from synesis.metrics import instantiate_metrics
from synesis.probes import get_probe
from synesis.utils import deep_update


def train(
    feature: str,
    dataset: str,
    task: str,
    item_format: str = "feature",
    task_config: Optional[dict] = None,
    feature_config: Optional[dict] = None,
    device: Optional[str] = None,
):
    """
    Train a downstream model.

    Args:
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        task: Name of the downstream task (needs to be supported by dataset).
        item_format: Format of the input data: ["raw", "feature"].
                     Defaults to "feature". If raw, feature is
                     extracted on-the-fly.
        task_config: Override certain values of the task configuration.
        device: Device to use for training (defaults to "cuda" if available).
    """

    if task_config:
        task_configs[task] = deep_update(task_configs[task], task_config)

    if feature_config:
        feature_configs[feature] = deep_update(feature_configs[feature], feature_config)

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="train",
        download=False,
        item_format=item_format,
    )
    val_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="validation",
        download=False,
        item_format=item_format,
    )

    assert task in train_dataset.tasks, f"Task {task} not available in {dataset}"

    if train_dataset[0][0].dim() == 3:
        # If item is 3D, this is a dataset that returns items with subitems
        # (e.g. for audio).
        if task_configs[task]["training"]["feature_aggregation"]:
            # If feature_aggreation, we'll wrap the dataset so that it returns
            # aggregated features
            aggregated_train = AggregateDataset(
                train_dataset, feature_extractor_name=feature
            )
            aggregated_val = AggregateDataset(
                val_dataset, feature_extractor_name=feature
            )
            del train_dataset, val_dataset
            train_dataset = aggregated_train
            val_dataset = aggregated_val
        else:
            # If not feature_aggregation, we'll wrap the dataset so that it behaves
            # as a subitem dataset
            wrapped_train = SubitemDataset(train_dataset)
            wrapped_val = SubitemDataset(val_dataset)
            del train_dataset, val_dataset
            train_dataset = wrapped_train
            val_dataset = wrapped_val

    dataloader = DataLoader(
        train_dataset,
        batch_size=task_configs[task]["training"]["batch_size"],
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=task_configs[task]["training"]["batch_size"],
        shuffle=False,
    )

    # if raw_data  (e.g. audio) is being returned from dataset,
    # extract features on-the-fly
    # (the AggregateDatset wrapper also computes features)
    if (
        item_format == "raw"
        and not task_configs[task]["training"]["feature_aggregation"]
    ):
        extractor = get_feature_extractor(feature)
        extractor.to(device)

    # train setup
    model = get_probe(
        model_type=task_configs[task]["model"]["type"],
        in_features=feature_configs[feature]["feature_dim"],
        n_outputs=len(train_dataset.label_encoder.classes_),
        **task_configs[task]["model"]["params"],
    ).to(device)
    criterion = task_configs[task]["training"]["criterion"]()
    optimizer_class = task_configs[task]["training"]["optimizer"]["class"]
    optimizer = optimizer_class(
        model.parameters(), **task_configs[task]["training"]["optimizer"]["params"]
    )

    val_metrics = instantiate_metrics(
        metric_configs=task_configs[task]["evaluation"]["metrics"],
        num_classes=len(train_dataset.label_encoder.classes_),
    )

    # train and validation loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    num_epochs = task_configs[task]["training"]["num_epochs"]
    patience = task_configs[task]["training"]["patience"]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for item, target in progress_bar:
            item = item.to(device)
            target = target.to(device)

            if (
                item_format == "raw"
                and not task_configs[task]["training"]["feature_aggregation"]
            ):
                with torch.no_grad():
                    item = extractor(item)
                    # if channels eaten up, unsqueeze
                    if item.dim() == 2:
                        item = item.unsqueeze(1)

            optimizer.zero_grad()
            output = model(item)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)

            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        model.eval()
        val_loss = 0
        val_outputs = []
        val_targets = []
        with torch.no_grad():
            for val_item, val_target in val_dataloader:
                val_item = val_item.to(device)
                val_target = val_target.to(device)

                if (
                    item_format == "raw"
                    and not task_configs[task]["training"]["feature_aggregation"]
                ):
                    with torch.no_grad():
                        item = extractor(item)

                val_output = model(val_item)
                val_loss += criterion(val_output, val_target).item()

                # Store outputs and targets for metric calculation
                val_outputs.append(val_output)
                val_targets.append(val_target)

        # Concatenate all outputs and targets
        val_outputs = torch.cat(val_outputs, dim=0)
        val_targets = torch.cat(val_targets, dim=0)

        # Calculate metrics
        val_metric_results = {}
        for metric_cfg, metric in zip(
            task_configs[task]["evaluation"]["metrics"], val_metrics
        ):
            metric = metric.to(device)
            val_metric_results[metric_cfg["name"]] = metric(val_outputs, val_targets)

        avg_val_loss = val_loss / len(val_dataloader)
        print(
            f"Epoch {epoch+1}/{num_epochs} -",
            f"Avg train loss: {avg_loss:.4f},",
            f"Avg val loss: {avg_val_loss:.4f}",
        )
        for name, value in val_metric_results.items():
            print(f"{name}: {value:.4f}")

        # Check if the validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the best model
    save_path = Path("ckpt") / "downstream" / f"{feature}_{dataset}_{task}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    return model


def evaluate(
    model: nn.Module,
    feature: str,
    dataset: str,
    task: str,
    item_format: str = "feature",
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
):
    """
    Evaluate a given trained downstream model.

    Args:
        model: Trained downstream model.
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        task: Name of the downstream task.
        item_format: Format of the input data: ["raw", "feature"].
                     Defaults to "feature". If raw, feature is
                     extracted on-the-fly.
        task_config: Override certain values of the task configuration.
        device: Device to use for evaluation (defaults to "cuda" if available).
    """

    if task_config:
        task_configs[task] = deep_update(task_configs[task], task_config)

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="test",
        download=False,
        item_format=item_format,
    )

    assert task in test_dataset.tasks

    metrics = instantiate_metrics(
        metric_configs=task_configs[task]["evaluation"]["metrics"],
        num_classes=len(test_dataset[0][1]),
    )

    if task_configs[task]["evaluation"]["feature_aggregation"]:
        dataloader = DataLoader(
            test_dataset,
            batch_size=task_configs[task]["evaluation"]["batch_size"],
            shuffle=False,
        )
    else:
        sampler = DynamicBatchSampler(
            dataset=test_dataset,
            batch_size=task_configs[task]["evaluation"]["batch_size"],
        )
        dataloader = DataLoader(
            test_dataset, batch_sampler=sampler, collate_fn=collate_packed_batch
        )

    if (
        item_format == "raw"
        and not task_configs[task]["evaluation"]["feature_aggregation"]
    ):
        extractor = get_feature_extractor(feature)
        extractor.to(device)

    model.eval()
    total_loss = 0
    test_outputs = []
    test_targets = []
    criterion = task_configs[task]["evaluation"]["criterion"]()

    with torch.no_grad():
        for item, target in dataloader:
            item = item.to(device)
            target = target.to(device)

            if item_format == "raw":
                with torch.no_grad():
                    item = extractor(item)

            output = model(item)
            total_loss += criterion(output, target).item()

            # Store outputs and targets for metric calculation
            test_outputs.append(output)
            test_targets.append(target)

    # Concatenate all outputs and targets
    test_outputs = torch.cat(test_outputs, dim=0)
    test_targets = torch.cat(test_targets, dim=0)

    # Calculate metrics
    test_metric_results = {}
    for metric_cfg, metric in zip(task_configs[task]["evaluation"]["metrics"], metrics):
        test_metric_results[metric_cfg["name"]] = metric(test_outputs, test_targets)

    avg_loss = total_loss / len(dataloader)
    print(f"Avg test loss: {avg_loss:.4f}")

    for name, value in test_metric_results.items():
        print(f"{name}: {value:.4f}")

    return test_metric_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a downstream model.")
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
        "--task",
        "-t",
        type=str,
        required=True,
        help="Task name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device to use for training.",
    )

    args = parser.parse_args()

    model = train(
        feature=args.feature,
        dataset=args.dataset,
        task=args.task,
        device=args.device,
    )

    results = evaluate(
        model=model,
        feature=args.feature,
        dataset=args.dataset,
        task=args.task,
        device=args.device,
    )
