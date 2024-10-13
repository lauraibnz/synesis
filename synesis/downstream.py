"""Methods for training and evaluating downstream models."""

import argparse
from typing import Optional

import torch
from metrics import instantiate_metrics
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.tasks import task_config as tc
from synesis.datasets.dataset_utils import get_dataset
from synesis.features.feature_utils import (
    DynamicBatchSampler,
    collate_packed_batch,
    get_pretrained_model,
)
from synesis.probes import get_probe
from synesis.utils import deep_update


def train(
    feature: str,
    dataset: str,
    task: str,
    item_format: str = "feature",
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
):
    """
    Train a downstream model.

    Args:
        feature: Name of the feature/embedding model (currently assumes embeddings
                 are already generated)
        dataset: Name of the dataset.
        task: Name of the downstream task (needs to be supported by dataset).
        item_format: Format of the input data: ["audio", "feature"].
                     Defaults to "feature". If audio, feature is
                     extracted on-the-fly.
        task_config: Override certain values of the task configuration.
        device: Device to use for training (defaults to "cuda" if available).
    """

    if task_config:
        tc[task] = deep_update(tc[task], task_config)

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="train",
        download=False,
    )
    val_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="validation",
        download=False,
    )

    assert task in train_dataset.tasks, f"Task {task} not available in {dataset}"

    if tc[task]["data"]["train"]["feature_aggregation"]:
        dataloader = DataLoader(
            train_dataset,
            batch_size=tc[task]["training"]["batch_size"],
            shuffle=True,
        )
    else:
        # use custom sampling for dynamic batching
        sampler = DynamicBatchSampler(
            dataset=dataset, batch_size=tc[task]["training"]["batch_size"]
        )
        dataloader = DataLoader(
            train_dataset, batch_sampler=sampler, collate_fn=collate_packed_batch
        )

    if tc[task]["data"]["evaluation"]["feature_aggregation"]:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=tc[task]["training"]["batch_size"],
            shuffle=False,
        )
    else:
        sampler = DynamicBatchSampler(
            dataset=val_dataset, batch_size=tc[task]["evaluation"]["batch_size"]
        )
        val_dataloader = DataLoader(
            val_dataset, batch_sampler=sampler, collate_fn=collate_packed_batch
        )

    # if audio is being returned from dataset, extract features on-the-fly
    if item_format == "audio":
        extractor = get_pretrained_model(feature)
        extractor.to(device)

    # train setup
    model = get_probe(
        model_type=tc[task]["model"]["type"],
        in_feaures=len(train_dataset[0][0][0]),
        n_outputs=len(train_dataset[0][1]),
        **tc[task]["model"]["params"],
    ).to(device)
    criterion = tc[task]["trainin"]["criterion"]
    optimizer_class = tc[task]["training"]["optimizer"]["class"]
    optimizer = optimizer_class(model.parameters(), **tc[task]["optimizer"]["params"])

    val_metrics = instantiate_metrics(
        metric_configs=tc[task]["evaluation"]["metrics"],
        num_classes=(train_dataset[0][1]),
    )

    # train and validation loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    num_epochs = tc[task]["training"]["num_epochs"]
    patience = tc[task]["training"]["patience"]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for item, target in progress_bar:
            item = item.to(device)
            target = target.to(device)

            if item_format == "audio":
                with torch.no_grad():
                    item = extractor(item)

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
        for metric_cfg, metric in zip(tc[task]["evaluation"]["metrics"], val_metrics):
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
        item_format: Format of the input data: ["audio", "feature"].
                     Defaults to "feature". If audio, feature is
                     extracted on-the-fly.
        task_config: Override certain values of the task configuration.
        device: Device to use for evaluation (defaults to "cuda" if available).
    """

    if task_config:
        tc[task] = deep_update(tc[task], task_config)

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="test",
        download=False,
    )

    assert task in test_dataset.tasks

    metrics = instantiate_metrics(
        metric_configs=tc[task]["evaluation"]["metrics"],
        num_classes=len(test_dataset[0][1]),
    )

    if tc[task]["evaluation"]["feature_aggregation"]:
        dataloader = DataLoader(
            test_dataset,
            batch_size=tc[task]["evaluation"]["batch_size"],
            shuffle=False,
        )
    else:
        sampler = DynamicBatchSampler(
            dataset=test_dataset, batch_size=tc[task]["evaluation"]["batch_size"]
        )
        dataloader = DataLoader(
            test_dataset, batch_sampler=sampler, collate_fn=collate_packed_batch
        )

    # if audio is being returned from dataset, extract features on-the-fly
    if item_format == "audio":
        extractor = get_pretrained_model(feature)
        extractor.to(device)

    model.eval()
    total_loss = 0
    test_outputs = []
    test_targets = []
    criterion = tc[task]["evaluation"]["criterion"]()

    with torch.no_grad():
        for item, target in dataloader:
            item = item.to(device)
            target = target.to(device)

            if item_format == "audio":
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
    for metric_cfg, metric in zip(tc[task]["evaluation"]["metrics"], metrics):
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
