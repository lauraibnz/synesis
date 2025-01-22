"""Methods for training and evaluating a model to predict the
transformation parameter of a given original and augmented
feature pair.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config.equivariance.parameters import configs as task_configs
from config.features import configs as feature_configs
from config.transforms import configs as transform_configs
from synesis.datasets.dataset_utils import SubitemDataset, get_dataset
from synesis.features.feature_utils import get_feature_extractor
from synesis.probes import get_probe
from synesis.transforms.transform_utils import get_transform
from synesis.utils import deep_update


def preprocess_batch(
    batch_raw_data,
    batch_targets,
    transform_obj,
    transform,
    feature_extractor,
    sample_rate,
    device,
):
    """Get transformed data, extract features from both the original and
    transformed data, and concatenate them for input to the model."""

    if any(
        tf in transform
        for tf in ["HueShift", "BrightnessShift", "SaturationShift", "JPEGCompression"]
    ):
        original_raw_data = batch_raw_data[:, 0].to(device)
        transformed_raw_data = batch_raw_data[:, 1].to(device)
        transform_params = batch_targets.to(device)

    elif "TimeStretch" in transform:
        # need to perform on gpu, item by item
        original_raw_data = batch_raw_data.to("cpu").numpy()
        transformed_raw_data = []
        transform_params = []
        for i in range(original_raw_data.shape[0]):
            transformed_item = transform_obj(
                original_raw_data[i][0], sample_rate=sample_rate
            )
            transform_param = transform_obj.parameters["rate"]

            # when slowed down, randomly decide an offset to crop
            # the original length from, to prevent overfiting/shortcutting
            # based on length of silence (happens when the track already
            # contains silence at one end)
            if len(transformed_item) > original_raw_data.shape[2]:
                offset = torch.randint(
                    0, len(transformed_item) - original_raw_data.shape[2], (1,)
                ).item()
                transformed_item = transformed_item[
                    offset : offset + original_raw_data.shape[2]
                ]
            # when sped up, figure out how much padding is needed, and
            # randomly decide how much to repeat pad from each side,
            # again to prevent learning the length of silence (the
            # left side will usually have speech)s
            elif len(transformed_item) < original_raw_data.shape[2]:
                pad = original_raw_data.shape[2] - len(transformed_item)
                left_pad = torch.randint(0, pad, (1,)).item()
                right_pad = pad - left_pad

                # Repeat pad left side
                left_audio = torch.tensor(
                    [
                        transformed_item[i % len(transformed_item)]
                        for i in range(left_pad)
                    ]
                )

                # Repeat pad right side
                right_audio = torch.tensor(
                    [
                        transformed_item[i % len(transformed_item)]
                        for i in range(
                            len(transformed_item) - right_pad, len(transformed_item)
                        )
                    ]
                )

                transformed_item = np.concatenate(
                    [left_audio, transformed_item, right_audio]
                )
            transformed_raw_data.append(torch.tensor(transformed_item))

            # map [0.5, 2.0] to [0, 1]
            transform_param = (transform_param - 0.5) / 1.5
            transform_params.append(transform_param)

        # make tensors and stack
        original_raw_data = batch_raw_data.to(device)
        transformed_raw_data = torch.stack(transformed_raw_data, dim=0).to(device)
        if transformed_raw_data.dim() == 2:
            transformed_raw_data = transformed_raw_data.unsqueeze(1)
        transform_params = torch.tensor(transform_params).to(device)
        assert original_raw_data.shape == transformed_raw_data.shape

    else:
        original_raw_data = batch_raw_data.to(device)
        transformed_raw_data = transform_obj(original_raw_data)

        # assert shape is the same after transformation
        assert original_raw_data.shape == transformed_raw_data.shape
        if "PitchShift" in transform:
            transform_params = [
                float(t_param)
                for t_param in transform_obj.transform_parameters["transpositions"]
            ]
            # map [0.5, 2.0] to [0, 1]
            transform_params = [(t_param - 0.5) / 1.5 for t_param in transform_params]

            # convert to tensor of shape [batch, 1, 1] and move to device
            transform_params = torch.tensor(transform_params).to(device)
        elif "AddWhiteNoise" in transform:
            transform_params = transform_obj.transform_parameters["snr_in_db"]
            # map [-30, 50] to [1, 0]
            transform_params = 1 - (transform_params + 30) / 80
            # convert to tensor of shape [batch, 1, 1] and move to device
            transform_params = torch.tensor(transform_params).to(device)
        else:
            # they will be of shape [batch, channel, 1], and on device
            transform_params = transform_obj.transform_parameters[
                f"{transform.lower()}_factors"
            ]

    # combine original and transformed data
    combined_raw_data = torch.cat([original_raw_data, transformed_raw_data], dim=0)

    with torch.no_grad():
        combined_features = feature_extractor(combined_raw_data)
        if combined_features.dim() == 2:
            combined_features = combined_features.unsqueeze(1)
        if combined_features.device != device:
            combined_features = combined_features.to(device)

    # currently, features are of shape (2b, c, t), where the first half of the
    # batch is originals, and the second is transformed. We need to split them
    # such that original feature 0 is concatenated with transformed 0, etc.
    original_features, transformed_features = torch.split(
        combined_features, batch_raw_data.size(0), dim=0
    )
    concat_features = torch.cat([original_features, transformed_features], dim=2)

    return concat_features, transform_params


def train(
    feature: str,
    dataset: str,
    transform: str,
    label: str,
    task: str,
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
    logging: bool = True,
):
    """Train a model to predict the transformation parameter of
    a given original and augmented feature pair. Does
    feature extraction on-the-fly.

    Args:
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        transform: Name of the transform.
        label: Name of the label.
        task: Name of the task.
        task_config: Override certain values of the task configuration.
        device: Device to use for training (defaults to "cuda" if available).
        logging: Whether to log to wandb.
    Returns:
        If logging is True, returns the wandb run path to the model artifact.
        Otherwise, returns the trained model.
    """
    feature_config = feature_configs.get(feature)
    transform_config = transform_configs.get(transform)
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )

    if logging:
        run_name = f"EQUI_PARA_{transform}_{label}_{dataset}_{feature}"
        wandb.init(
            project="synesis",
            name=run_name,
            config={
                "feature": feature,
                "feature_config": feature_config,
                "dataset": dataset,
                "task": task,
                "task_config": task_config,
                "label": label,
            },
        )
        artifact = wandb.Artifact(run_name, type="model", metadata={"task": task})

    if task_config["training"].get("feature_aggregation") or task_config[
        "evaluation"
    ].get("feature_aggregation"):
        raise NotImplementedError(
            "Feature aggregation is not currently implemented for transform prediction."
        )

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = get_dataset(
        name=dataset,
        feature=feature,
        transform=transform,
        label=label,
        split="train",
        download=False,
        item_format="raw",
    )
    val_dataset = get_dataset(
        name=dataset,
        feature=feature,
        transform=transform,
        label=label,
        split="validation",
        download=False,
        item_format="raw",
    )

    # If dataset returns subitems per item, need to wrap it
    if dataset != "ImageNet" and train_dataset[0][0].dim() == 3:
        wrapped_train = SubitemDataset(train_dataset)
        wrapped_val = SubitemDataset(val_dataset)
        del train_dataset, val_dataset
        train_dataset = wrapped_train
        val_dataset = wrapped_val

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

    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=task_config["training"]["batch_size"]
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=task_config["training"]["batch_size"]
    )

    model = get_probe(
        model_type=task_config["model"]["type"],
        in_features=feature_config["feature_dim"] * 2,
        n_outputs=1,  # currently only predicting one parameter
        **task_config["model"]["params"],
    ).to(device)

    criterion = task_configs[task]["training"]["criterion"]()
    optimizer_class = task_configs[task]["training"]["optimizer"]["class"]
    optimizer = optimizer_class(
        model.parameters(),
        **task_configs[task]["training"]["optimizer"]["params"],
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    num_epochs = task_config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_raw_data, batch_targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            # prepare data for equivariance training
            concat_features, transform_params = preprocess_batch(
                batch_raw_data=batch_raw_data,
                batch_targets=batch_targets,
                transform_obj=transform_obj,
                transform=transform,
                feature_extractor=feature_extractor,
                sample_rate=feature_config.get("sample_rate", None),
                device=device,
            )

            optimizer.zero_grad()
            predicted_params = model(concat_features)
            if len(predicted_params.shape) == 2:
                predicted_params = predicted_params.squeeze(1)
            loss = criterion(predicted_params, transform_params)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if logging:
                wandb.log({"train/loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_raw_data, batch_targets in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                # prepare data for equivariance training
                concat_features, transform_params = preprocess_batch(
                    batch_raw_data=batch_raw_data,
                    batch_targets=batch_targets,
                    transform_obj=transform_obj,
                    transform=transform,
                    sample_rate=feature_config.get("sample_rate", None),
                    feature_extractor=feature_extractor,
                    device=device,
                )

                predicted_params = model(concat_features)
                if len(predicted_params.shape) == 2:
                    predicted_params = predicted_params.squeeze(1)
                loss = criterion(predicted_params, transform_params)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
            + f"Val Loss: {avg_val_loss:.4f}"
        )

        if logging:
            wandb.log({"val/loss": avg_val_loss})

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= task_config["training"]["patience"]:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    save_path = Path("ckpt") / "EQUI" / "PARA" / transform / dataset / f"{feature}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    if logging:
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)
        wandb_path = wandb.run.path + "/" + artifact.name
        wandb.finish()
        return wandb_path

    return model


def evaluate(
    model: Union[nn.Module, str],
    feature: str,
    dataset: str,
    transform: str,
    label: str,
    task: str,
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
    logging: bool = True,
):
    """
    Evaluate a given trained model for predicting transformation parameters.

    Args:
        model: Trained model for predicting transformation parameters.
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        transform: Name of the transform (factor of variation).
        task: Name of the task.
        task_config: Override certain values of the task configuration.
        device: Device to use for evaluation (defaults to "cuda" if available).
        logging: Whether to log to wandb.
    Returns:
        Dictionary of evaluation metrics.
    """

    if isinstance(model, str):
        # resume wandb run
        entity, project, run_id, model_name = model.split("/")
        if logging:
            wandb.init(project=project, entity=entity, id=run_id, resume="allow")

    feature_config = feature_configs.get(feature)
    transform_config = transform_configs.get(transform)
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )

    test_dataset = get_dataset(
        name=dataset,
        feature=feature,
        label=label,
        transform=transform,
        split="test",
        download=False,
        item_format="raw",
    )

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(model, str):
        # Load model from wandb artifact
        model_wandb_path = f"{entity}/{project}/{model_name}"
        artifact_name = (
            f"{model_wandb_path}:latest" if ":" not in model_name else model_name
        )
        artifact = wandb.Api().artifact(f"{entity}/{project}/{artifact_name}")
        artifact_dir = artifact.download()
        model = get_probe(
            model_type=task_config["model"]["type"],
            in_features=feature_config["feature_dim"] * 2,
            n_outputs=1,  # currently only predicting one parameter
            **task_config["model"]["params"],
        )
        model.load_state_dict(torch.load(Path(artifact_dir) / f"{feature}.pt"))
        os.remove(Path(artifact_dir) / f"{feature}.pt")

    model.to(device)

    if task_config["training"].get("feature_aggregation") or task_config[
        "evaluation"
    ].get("feature_aggregation"):
        raise NotImplementedError(
            "Feature aggregation is not currently implemented for transform prediction."
        )

    # If dataset returns subitems per item, need to wrap it
    if dataset != "ImageNet" and test_dataset[0][0].dim() == 3:
        wrapped_test = SubitemDataset(test_dataset)
        del test_dataset
        test_dataset = wrapped_test

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

    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=task_config["evaluation"]["batch_size"]
    )

    model.eval()
    total_loss = 0
    all_predicted_params = []
    all_true_params = []

    criterion = task_configs[task]["training"]["criterion"]()

    with torch.no_grad():
        for batch_raw_data, batch_targets in tqdm(test_loader, desc="Evaluating"):
            # prepare data for equivariance prediction
            concat_features, transform_params = preprocess_batch(
                batch_raw_data=batch_raw_data,
                batch_targets=batch_targets,
                transform_obj=transform_obj,
                transform=transform,
                sample_rate=feature_config.get("sample_rate", None),
                feature_extractor=feature_extractor,
                device=device,
            )

            predicted_params = model(concat_features)
            if len(predicted_params.shape) == 2:
                predicted_params = predicted_params.squeeze(1)
            loss = criterion(predicted_params, transform_params)

            total_loss += loss.item()

            all_predicted_params.append(predicted_params.cpu())
            all_true_params.append(transform_params.cpu())

    avg_loss = total_loss / len(test_loader)
    print(f"Average test loss: {avg_loss:.4f}")

    # Concatenate all predictions and true parameters
    all_predicted_params = torch.cat(all_predicted_params, dim=0)
    all_true_params = torch.cat(all_true_params, dim=0)

    # Calculate additional metrics
    mse = nn.MSELoss()(all_predicted_params, all_true_params).item()
    mae = nn.L1Loss()(all_predicted_params, all_true_params).item()

    # Calculate R-squared
    ss_tot = torch.sum((all_true_params - all_true_params.mean()) ** 2)
    ss_res = torch.sum((all_true_params - all_predicted_params) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    if logging:
        # Create a table for the evaluation metrics
        metrics_table = wandb.Table(columns=["Metric", "Value"])
        metrics_table.add_data("Average Loss", avg_loss)
        metrics_table.add_data("Mean Squared Error", mse)
        metrics_table.add_data("Mean Absolute Error", mae)
        metrics_table.add_data("R-squared", r_squared.item())

        # Log the table to wandb
        wandb.log({"evaluation_metrics": metrics_table})
        wandb.finish()

    return {"avg_loss": avg_loss, "mse": mse, "mae": mae, "r_squared": r_squared.item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a transform param prediction model."
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
        "--nolog",
        action="store_true",
        help="Do not log to wandb.",
    )

    args = parser.parse_args()

    model = train(
        feature=args.feature,
        dataset=args.dataset,
        transform=args.transform,
        label=args.label,
        task=args.task,
        device=args.device,
        logging=not args.nolog,
    )

    results = evaluate(
        model=model,
        feature=args.feature,
        dataset=args.dataset,
        transform=args.transform,
        label=args.label,
        task=args.task,
        device=args.device,
        logging=not args.nolog,
    )
