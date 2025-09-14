"""Methods for evaluating concatenated features from two different feature extractors.

This script trains a probe on concatenated embeddings from two features and compares
the performance to using just the first feature alone to measure the benefit of
feature combination.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import wandb
from config.features import configs as feature_configs
from config.informativeness.downstream import configs as task_configs
from synesis.datasets.dataset_utils import AggregateDataset, SubitemDataset, get_dataset
from synesis.features.feature_utils import get_feature_extractor
from synesis.metrics import instantiate_metrics
from synesis.probes import get_probe
from synesis.utils import deep_update, get_artifact, get_metric_from_wandb, get_wandb_config


def concatenate_features(features1, features2):
    """Concatenate features from two extractors, handling different temporal dimensions."""
    unsqueeze = features1.dim() == 4 or features2.dim() == 4
    # Remove singleton channel dimensions if present
    if unsqueeze:
        assert features1.shape[1] == 1, "features1 should have shape [B, 1, C] or [B, 1, T]"
        features1 = features1.squeeze(1)  # [B, 1, C] -> [B, C] or [B, 1, T] -> [B, T]
        assert features2.shape[1] == 1, "features2 should have shape [B, 1, C] or [B, 1, T]"
        features2 = features2.squeeze(1)  # [B, 1, C] -> [B, C] or [B, 1, T] -> [B, T]
    
    # Handle temporal dimension broadcasting
    if features1.dim() == 3 and features2.dim() == 2:
        # features1: [B, C1, T], features2: [B, C2] -> broadcast features2 to [B, C2, T]
        B, C2 = features2.shape
        T = features1.shape[2]
        features2 = features2.unsqueeze(2).expand(B, C2, T)  # [B, C2, T]
    elif features1.dim() == 2 and features2.dim() == 3:
        # features1: [B, C1], features2: [B, C2, T] -> broadcast features1 to [B, C1, T]
        B, C1 = features1.shape
        T = features2.shape[2]
        features1 = features1.unsqueeze(2).expand(B, C1, T)  # [B, C1, T]
    
    # Now concatenate along feature dimension (dim=1)
    concatenated = torch.cat([features1, features2], dim=1)  # [B, C1+C2] or [B, C1+C2, T]

    if unsqueeze:
        concatenated = concatenated.unsqueeze(1)
    
    return concatenated


def train_concatenated(
    feature1: str,
    feature2: str,
    dataset: str,
    task: str,
    label: str,
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
    logging: bool = False,
):
    """Train a model on concatenated features from two feature extractors."""
    
    feature1_config = feature_configs[feature1]
    feature2_config = feature_configs[feature2]
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )
    
    if logging:
        run_name = f"CONCAT_{task}_{dataset}_{label}_{feature1}_{feature2}"
        wandb_config = get_wandb_config()
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=run_name,
            config={
                "feature1": feature1,
                "feature1_config": feature1_config,
                "feature2": feature2,
                "feature2_config": feature2_config,
                "dataset": dataset,
                "task": task,
                "task_config": task_config,
                "label": label,
            },
        )
        artifact = wandb.Artifact(run_name, type="model", metadata={"task": task})

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load datasets for both features
    train_dataset1 = get_dataset(
        name=dataset,
        feature=feature1,
        label=label,
        split="train",
        download=False,
        item_format="feature",
    )
    val_dataset1 = get_dataset(
        name=dataset,
        feature=feature1,
        label=label,
        split="validation",
        download=False,
        item_format="feature",
    )
    
    train_dataset2 = get_dataset(
        name=dataset,
        feature=feature2,
        label=label,
        split="train",
        download=False,
        item_format="feature",
    )
    val_dataset2 = get_dataset(
        name=dataset,
        feature=feature2,
        label=label,
        split="validation",
        download=False,
        item_format="feature",
    )

    # Handle subitems if needed
    if train_dataset1[0][0].dim() > 2 and dataset != "ImageNet":
        if task_config["training"]["feature_aggregation"]:
            train_dataset1 = AggregateDataset(train_dataset1, feature_extractor_name=feature1)
            val_dataset1 = AggregateDataset(val_dataset1, feature_extractor_name=feature1)
            train_dataset2 = AggregateDataset(train_dataset2, feature_extractor_name=feature2)
            val_dataset2 = AggregateDataset(val_dataset2, feature_extractor_name=feature2)
        else:
            train_dataset1 = SubitemDataset(train_dataset1)
            val_dataset1 = SubitemDataset(val_dataset1)
            train_dataset2 = SubitemDataset(train_dataset2)
            val_dataset2 = SubitemDataset(val_dataset2)


    sampler1 = RandomSampler(train_dataset1, generator=torch.Generator().manual_seed(42))
    sampler2 = RandomSampler(train_dataset2, generator=torch.Generator().manual_seed(42))

    train_loader1 = DataLoader(
        train_dataset1,
        batch_size=task_config["training"]["batch_size"],
        sampler=sampler1,
    )
    
    train_loader2 = DataLoader(
        train_dataset2,
        batch_size=task_config["training"]["batch_size"],
        sampler=sampler2,
    )

    val_loader1 = DataLoader(
        val_dataset1,
        batch_size=task_config["training"]["batch_size"],
        shuffle=False,
    )
    val_loader2 = DataLoader(
        val_dataset2,
        batch_size=task_config["training"]["batch_size"],
        shuffle=False,
    )

    # Determine output dimensions
    if task_config["model"]["type"] == "transcriber":
        n_outputs = 88
    else:
        n_outputs = (
            1
            if task_config["model"]["type"] == "regressor"
            else len(train_dataset1.label_encoder.classes_)
        )

    # Get sample items to determine concatenated feature dimensions
    sample_item1, _ = train_dataset1[0]
    sample_item2, _ = train_dataset2[0]
    
    # Determine if we need temporal pooling based on sample dimensions
    sample_concat = concatenate_features(sample_item1, sample_item2)
    use_temporal_pooling = sample_concat.dim() == 3  # True if result has temporal dimension

    total_in_features = sample_concat.shape[1]

    model = get_probe(
        model_type=task_config["model"]["type"],
        in_features=total_in_features,
        n_outputs=n_outputs,
        use_temporal_pooling=use_temporal_pooling,
        **task_config["model"]["params"],
    ).to(device)
    
    criterion = task_config["training"]["criterion"]()
    optimizer_class = task_config["training"]["optimizer"]["class"]
    optimizer = optimizer_class(
        model.parameters(), **task_config["training"]["optimizer"]["params"]
    )

    val_metrics = instantiate_metrics(
        metric_configs=task_config["evaluation"]["metrics"],
        num_classes=n_outputs,
    )

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    num_epochs = task_config["training"]["num_epochs"]
    patience = task_config["training"]["patience"]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Zip the two data loaders together
        progress_bar = tqdm(
            zip(train_loader1, train_loader2), 
            desc=f"Epoch {epoch+1}/{num_epochs}",
            total=min(len(train_loader1), len(train_loader2))
        )

        for (item1, target1), (item2, target2) in progress_bar:
            # Ensure targets match (they should, but verify)
            assert torch.equal(target1, target2), "Targets from both datasets must match"
            
            item1 = item1.to(device)
            item2 = item2.to(device)
            target = target1.to(device)

            # Concatenate features
            concatenated_item = concatenate_features(item1, item2)
            
            optimizer.zero_grad()
            output = model(concatenated_item)
            if len(output.shape) == 2 and n_outputs == 1:
                output = output.squeeze(1)

            if isinstance(criterion, nn.BCEWithLogitsLoss):
                target_for_loss = target.float()
            else:
                target_for_loss = target

            loss = criterion(output, target_for_loss)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if logging:
                wandb.log({"train/loss": loss.item()})

        # Validation
        model.eval()
        val_loss = 0
        val_metric_results = {}
        
        with torch.no_grad():
            val_progress = tqdm(
                zip(val_loader1, val_loader2),
                desc=f"Epoch {epoch+1}/{num_epochs} - Validation",
                total=min(len(val_loader1), len(val_loader2))
            )
            
            for (item1, target1), (item2, target2) in val_progress:
                # Ensure targets match
                assert torch.equal(target1, target2), "Targets from both datasets must match"
                
                item1 = item1.to(device)
                item2 = item2.to(device)
                target = target1.to(device)

                # Concatenate features
                concatenated_item = concatenate_features(item1, item2)
                
                val_output = model(concatenated_item)
                if len(val_output.shape) == 2 and n_outputs == 1:
                    val_output = val_output.squeeze(1)

                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    target_for_loss = target.float()
                else:
                    target_for_loss = target

                val_loss += criterion(val_output, target_for_loss).item()

                if task_config["model"]["type"] == "regressor":
                    target_for_metrics = target_for_loss
                else:
                    target_for_metrics = target.long()

                for metric_cfg, metric in zip(
                    task_config["evaluation"]["metrics"], val_metrics
                ):
                    metric = metric.to(device)
                    if task_config["model"]["type"] == "transcriber":
                        if metric_cfg["name"] == "NoteMetrics":
                            threshold = 0.1
                            val_output = (val_output > threshold).int()
                        elif metric_cfg["name"] == "F1":
                            threshold = 0.3
                            val_output = (val_output > threshold).int()
                    metric.update(val_output, target_for_metrics)

        # Calculate validation metrics
        avg_val_loss = val_loss / min(len(val_loader1), len(val_loader2))
        for metric_cfg, metric in zip(
            task_config["evaluation"]["metrics"], val_metrics
        ):
            val_metric_results[metric_cfg["name"]] = metric.compute().item()
            metric.reset()
            
        print(
            f"Epoch {epoch+1}/{num_epochs} -",
            f"Avg train loss: {avg_loss:.4f},",
            f"Avg val loss: {avg_val_loss:.4f}",
        )
        for name, value in val_metric_results.items():
            print(f"{name}: {value:.4f}")

        if logging:
            wandb.log(
                {
                    "val/loss": avg_val_loss,
                    **{
                        f"val/{name}": value
                        for name, value in val_metric_results.items()
                    },
                }
            )

        # Check for improvement
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

    # Save the model
    save_path = Path("ckpt") / "CONCAT" / task / dataset / f"{feature1}_{feature2}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    if logging:
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)
        wandb_path = wandb.run.path + "/" + artifact.name
        wandb.finish()
        return wandb_path
    
    return model


def evaluate_concatenated(
    model: Union[nn.Module, str],
    feature1: str,
    feature2: str,
    dataset: str,
    task: str,
    label: str,
    baseline_run,
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
    logging: bool = False,
):
    """Evaluate concatenated features and compare to feature1 alone."""
    
    feature1_config = feature_configs[feature1]
    feature2_config = feature_configs[feature2]
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )

    # Resume the wandb run if model is from wandb (like informativeness script)
    if isinstance(model, str):
        entity, project, run_id, model_name = model.split("/")
        if logging:
            wandb.init(project=project, entity=entity, id=run_id, resume="allow")

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test datasets for both features
    test_dataset1 = get_dataset(
        name=dataset,
        feature=feature1,
        label=label,
        split="test",
        download=False,
        item_format="feature",
    )
    test_dataset2 = get_dataset(
        name=dataset,
        feature=feature2,
        label=label,
        split="test",
        download=False,
        item_format="feature",
    )

    # Handle subitems if needed
    if test_dataset1[0][0].dim() > 2 and dataset != "ImageNet":
        if task_config["evaluation"]["feature_aggregation"]:
            test_dataset1 = AggregateDataset(test_dataset1, feature_extractor_name=feature1)
            test_dataset2 = AggregateDataset(test_dataset2, feature_extractor_name=feature2)
        else:
            test_dataset1 = SubitemDataset(test_dataset1)
            test_dataset2 = SubitemDataset(test_dataset2)

    test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)
    test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)

    # Determine model architecture
    if task_config["model"]["type"] == "transcriber":
        n_outputs = 88
    else:
        n_outputs = (
            1
            if task_config["model"]["type"] == "regressor"
            else len(test_dataset1.label_encoder.classes_)
        )

    # For concatenated model evaluation, we use the concatenated dimensions
    sample_item1, _ = test_dataset1[0]
    sample_item2, _ = test_dataset2[0]
    sample_concat = concatenate_features(sample_item1.unsqueeze(0), sample_item2.unsqueeze(0)).squeeze(0)
    
    if sample_concat.dim() == 1:
        total_in_features = sample_concat.shape[0]
    else:
        total_in_features = sample_concat.shape[1]
    
    use_temporal_pooling = sample_concat.dim() == 3

    if isinstance(model, str):
        # Load concatenated model from wandb artifact
        artifact = get_artifact(model)
        artifact_dir = artifact.download()
        
        model = get_probe(
            model_type=task_config["model"]["type"],
            in_features=total_in_features,
            n_outputs=n_outputs,
            use_temporal_pooling=use_temporal_pooling,
            **task_config["model"]["params"],
        )
        model.load_state_dict(torch.load(Path(artifact_dir) / f"{feature1}_{feature2}.pt"))
        os.remove(Path(artifact_dir) / f"{feature1}_{feature2}.pt")

    model.to(device)
    model.eval()

    # Evaluate concatenated model
    total_loss = 0
    criterion = task_config["evaluation"]["criterion"]()
    
    metrics = instantiate_metrics(
        metric_configs=task_config["evaluation"]["metrics"],
        num_classes=n_outputs,
    )

    with torch.no_grad():
        progress_bar = tqdm(
            zip(test_loader1, test_loader2),
            desc="Evaluating concatenated",
            total=min(len(test_loader1), len(test_loader2))
        )
        
        for (item1, target1), (item2, target2) in progress_bar:
            # Ensure targets match
            assert torch.equal(target1, target2), "Targets from both datasets must match"
            
            item1 = item1.to(device)
            item2 = item2.to(device)
            target = target1.to(device)

            # Concatenate features for the concatenated model
            concatenated_item = concatenate_features(item1, item2)
            
            output = model(concatenated_item)
            if len(output.shape) == 2 and n_outputs == 1:
                output = output.squeeze(1)

            if isinstance(criterion, nn.BCEWithLogitsLoss):
                target_for_loss = target.float()
            else:
                target_for_loss = target

            total_loss += criterion(output, target_for_loss).item()

            if task_config["model"]["type"] == "regressor":
                target_for_metrics = target_for_loss
            else:
                target_for_metrics = target.long()

            for metric_cfg, metric in zip(task_config["evaluation"]["metrics"], metrics):
                metric = metric.to(device)
                if task_config["model"]["type"] == "transcriber":
                    if metric_cfg["name"] == "NoteMetrics":
                        threshold = 0.1
                        output = (output > threshold).int()
                    elif metric_cfg["name"] == "F1":
                        threshold = 0.3
                        output = (output > threshold).int()
                metric.update(output, target_for_metrics)

    # Calculate concatenated metrics
    avg_loss = total_loss / min(len(test_loader1), len(test_loader2))
    concat_metric_results = {}
    
    for metric_cfg, metric in zip(task_config["evaluation"]["metrics"], metrics):
        concat_metric_results[metric_cfg["name"]] = metric.compute().item()
        metric.reset()
    
    print(f"Concatenated model - Avg test loss: {avg_loss:.4f}")
    for name, value in concat_metric_results.items():
        print(f"Concatenated {name}: {value:.4f}")

    # Get original metrics from the baseline informativeness run
    print(f"\nEvaluating original {feature1}-only model on test set...")
    
    # Use the baseline_run that was already found in main (with the newest one selected)
    wandb_config = get_wandb_config()
    entity = wandb_config["entity"]
    project = wandb_config["project"]
    original_run_name = f"INFO_DOWN_{task}_{dataset}_{label}_{feature1}"
    
    # Load and evaluate the original feature1-only model
    original_artifact = get_artifact(f"{entity}/{project}/{baseline_run.id}/{original_run_name}")
    original_artifact_dir = original_artifact.download()
    
    # Get feature1 dimensions for the original model
    sample_item1, _ = test_dataset1[0]
    if sample_item1.dim() == 1:
        original_in_features = sample_item1.shape[0]
    else:
        original_in_features = sample_item1.shape[1]
    
    original_use_temporal_pooling = sample_item1.dim() == 3
    
    # Create the original model architecture
    original_model = get_probe(
        model_type=task_config["model"]["type"],
        in_features=original_in_features,
        n_outputs=n_outputs,
        use_temporal_pooling=original_use_temporal_pooling,
        **task_config["model"]["params"],
    ).to(device)
    
    # Load the original model weights
    original_model.load_state_dict(torch.load(Path(original_artifact_dir) / f"{feature1}.pt", weights_only=True))
    original_model.eval()
    
    # Evaluate original model on feature1 only
    original_total_loss = 0
    original_metrics_objs = instantiate_metrics(
        metric_configs=task_config["evaluation"]["metrics"],
        num_classes=n_outputs,
    )
    
    with torch.no_grad():
        for (item1, target1), (item2, target2) in tqdm(
            zip(test_loader1, test_loader2), 
            desc="Evaluating original feature1-only model",
            total=min(len(test_loader1), len(test_loader2))
        ):
            # Use only feature1 for the original model
            item1 = item1.to(device)
            target = target1.to(device)
            
            output = original_model(item1)
            if len(output.shape) == 2 and n_outputs == 1:
                output = output.squeeze(1)

            if isinstance(criterion, nn.BCEWithLogitsLoss):
                target_for_loss = target.float()
            else:
                target_for_loss = target

            original_total_loss += criterion(output, target_for_loss).item()

            if task_config["model"]["type"] == "regressor":
                target_for_metrics = target_for_loss
            else:
                target_for_metrics = target.long()

            for metric_cfg, metric in zip(task_config["evaluation"]["metrics"], original_metrics_objs):
                metric = metric.to(device)
                if task_config["model"]["type"] == "transcriber":
                    if metric_cfg["name"] == "NoteMetrics":
                        threshold = 0.1
                        output = (output > threshold).int()
                    elif metric_cfg["name"] == "F1":
                        threshold = 0.3
                        output = (output > threshold).int()
                metric.update(output, target_for_metrics)
    
    # Calculate original metrics
    original_avg_loss = original_total_loss / min(len(test_loader1), len(test_loader2))
    original_metrics = {}
    for metric_cfg, metric in zip(task_config["evaluation"]["metrics"], original_metrics_objs):
        original_metrics[metric_cfg["name"]] = metric.compute().item()
        metric.reset()
    
    print(f"Original {feature1}-only model - Avg test loss: {original_avg_loss:.4f}")
    for name, value in original_metrics.items():
        print(f"Original {name}: {value:.4f}")
    
    # Clean up
    os.remove(Path(original_artifact_dir) / f"{feature1}.pt")

    # Calculate differences (concatenated - original)
    diff_metrics = {}
    for name in concat_metric_results.keys():
        if name in original_metrics:
            diff = concat_metric_results[name] - original_metrics[name]
            diff_metrics[f"diff_{name}"] = diff
            print(f"{name} improvement: {diff:.4f}")

    # Prepare results for logging
    results = {
        "avg_loss": avg_loss,
        **concat_metric_results,
        **diff_metrics,
    }

    if logging:
        # Log test metrics like the informativeness script does
        wandb.log({
            **{f"test/{name}": value for name, value in concat_metric_results.items()},
            **diff_metrics,
        })
        
        # Create a table for the evaluation metrics (like informativeness script)
        metrics_table = wandb.Table(columns=["Metric", "Value"])
        metrics_table.add_data("Average Test Loss", avg_loss)
        for name, value in concat_metric_results.items():
            metrics_table.add_data(name, value)
        
        # Add difference metrics to the table
        for name, value in diff_metrics.items():
            metrics_table.add_data(name, value)
        
        # Log the table to wandb
        wandb.log({"evaluation_metrics": metrics_table})
        wandb.finish()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate concatenated features from two extractors."
    )
    parser.add_argument(
        "--feature1",
        "-f1",
        type=str,
        required=True,
        help="First feature name.",
    )
    parser.add_argument(
        "--feature2",
        "-f2",
        type=str,
        required=True,
        help="Second feature name.",
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
        "--label",
        "-l",
        type=str,
        required=True,
        help="Factor of variation or label to predict.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device to use for training.",
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

    # First, check if the baseline informativeness run exists (like in disentanglement script)
    wandb_config = get_wandb_config()
    entity = wandb_config["entity"]
    project = wandb_config["project"]
    baseline_run_name = f"INFO_DOWN_{args.task}_{args.dataset}_{args.label}_{args.feature1}"
    
    print(f"Looking for baseline run: {baseline_run_name}")
    api = wandb.Api()
    wandb_runs = api.runs(f"{entity}/{project}")
    baseline_run = None
    for run in wandb_runs:
        if run.name == baseline_run_name:
            baseline_run = run

    print(f"Found baseline run: {run.id}")
    
    if baseline_run is None:
        raise ValueError(f"Baseline run {baseline_run_name} not found. Please run informativeness evaluation for {args.feature1} first.")

    # Train concatenated model
    model = train_concatenated(
        feature1=args.feature1,
        feature2=args.feature2,
        dataset=args.dataset,
        task=args.task,
        label=args.label,
        device=args.device,
        logging=not args.nolog,
    )

    # Evaluate and compare
    results = evaluate_concatenated(
        model=model,
        feature1=args.feature1,
        feature2=args.feature2,
        dataset=args.dataset,
        task=args.task,
        label=args.label,
        baseline_run=baseline_run,
        device=args.device,
        batch_size=args.batch_size,
        logging=not args.nolog,
    ) 