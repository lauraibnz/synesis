"""Methods for training and evaluating a model to predict the
transformed feature given an original feature and
a transformation parameter.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config.equivariance.features import configs as task_configs
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
    if transform in ["HueShift", "BrightnessShift", "SaturationShift"]:
        original_raw_data = batch_raw_data[:, 0].to(device)
        transformed_raw_data = batch_raw_data[:, 1].to(device)
        transform_params = batch_targets
        if len(transform_params.shape) == 1:
            transform_params = transform_params.unsqueeze(1)
        if len(transform_params.shape) == 2:
            transform_params = transform_params.unsqueeze(2)
        transform_params = transform_params.to(device)

    elif transform == "TimeStretch":
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
        transform_params = (
            torch.tensor(transform_params).unsqueeze(1).unsqueeze(2).to(device)
        )
        assert original_raw_data.shape == transformed_raw_data.shape

    else:
        original_raw_data = batch_raw_data.to(device)
        transformed_raw_data = transform_obj(original_raw_data)
        # assert shape is the same after transformation
        assert original_raw_data.shape == transformed_raw_data.shape
        # get transformation parameters that were actually applied to batch
        if transform == "PitchShift":
            transform_params = [
                float(t_param)
                for t_param in transform_obj.transform_parameters["transpositions"]
            ]
            # map [0.5, 2.0] to [0, 1]
            transform_params = [(t_param - 0.5) / 1.5 for t_param in transform_params]

            # convert to tensor of shape [batch, 1, 1] and move to device
            transform_params = (
                torch.tensor(transform_params).unsqueeze(1).unsqueeze(1).to(device)
            )
        elif transform == "AddWhiteNoise":
            transform_params = transform_obj.transform_parameters["snr_in_db"]
            # map [-30, 50] to [1, 0]
            transform_params = 1 - (transform_params + 30) / 80
            # convert to tensor of shape [batch, 1, 1] and move to device
            transform_params = (
                torch.tensor(transform_params).unsqueeze(1).unsqueeze(1).to(device)
            )
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

    # split the combined features back into original and transformed features
    original_features, transformed_features = torch.split(
        combined_features, batch_raw_data.shape[0], dim=0
    )

    return original_features, transformed_features, transform_params


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
    """Train a model to predict the transformed feature given
    an original feature and a transformation parameter. Does
    feature extraction on-the-fly.

    Args:
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        transform: Name of the transform.
        label: Name of the label (factor of variation).
        task: Name of the downstream task.
        task_config: Override certain values of the task configuration.
        device: Device to use for training (defaults to "cuda" if available).
        logging: Whether to log the model to wandb.
    Returns:
        If logging is True, returns the wandb run path to the model artifact.
        Otherwise, returns trained model.
    """
    feature_config = feature_configs.get(feature)
    transform_config = transform_configs.get(transform)
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )

    if logging:
        run_name = f"EMB_EQUI_FEAT_{task}_{transform}_{label}_{dataset}_{feature}"
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
        label=label,
        transform=transform,
        split="train",
        download=False,
        item_format="raw",
    )
    val_dataset = get_dataset(
        name=dataset,
        feature=feature,
        label=label,
        transform=transform,
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
    if transform == "PitchShift" or transform == "AddWhiteNoise":
        transform_obj = get_transform(
            transform_config,
            sample_rate=feature_config["sample_rate"],
        )
    elif dataset == "ImageNet":
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
        in_features=feature_config["feature_dim"],
        emb_param=True,
        n_outputs=feature_config["feature_dim"],
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
            original_features, transformed_features, transform_params = (
                preprocess_batch(
                    batch_raw_data=batch_raw_data,
                    batch_targets=batch_targets,
                    transform_obj=transform_obj,
                    transform=transform,
                    feature_extractor=feature_extractor,
                    sample_rate=feature_config.get("sample_rate", None),
                    device=device,
                )
            )

            optimizer.zero_grad()
            if transform_params.dim() == 3:
                transform_params = transform_params.squeeze(1)
            predicted_features = model(original_features, param=transform_params)
            if predicted_features.dim() == 2 and original_features.dim() == 3:
                predicted_features = predicted_features.unsqueeze(1)
            loss = criterion(original_features, predicted_features)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if logging:
                wandb.log({"train/loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        total_l2_distance = 0
        total_cosine_distance = 0
        with torch.no_grad():
            for batch_raw_data, batch_targets in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                # prepare data for equivariance training
                original_features, transformed_features, transform_params = (
                    preprocess_batch(
                        batch_raw_data=batch_raw_data,
                        batch_targets=batch_targets,
                        transform_obj=transform_obj,
                        transform=transform,
                        feature_extractor=feature_extractor,
                        sample_rate=feature_config.get("sample_rate", None),
                        device=device,
                    )
                )

                if transform_params.dim() == 3:
                    transform_params = transform_params.squeeze(1)
                predicted_features = model(original_features, param=transform_params)
                if predicted_features.dim() == 2 and original_features.dim() == 3:
                    predicted_features = predicted_features.unsqueeze(1)
                loss = criterion(original_features, predicted_features)

                total_val_loss += loss.item()

                # Compute L2 distance
                l2_distance = F.mse_loss(
                    predicted_features, transformed_features, reduction="sum"
                )
                total_l2_distance += l2_distance.item()

                # Compute cosine distance
                cosine_distance = (
                    1
                    - F.cosine_similarity(
                        predicted_features, transformed_features, dim=1
                    )
                    .sum()
                    .item()
                )
                total_cosine_distance += cosine_distance

            avg_val_loss = total_val_loss / len(val_loader)
            avg_l2_distance = total_l2_distance / len(val_loader.dataset)
            avg_cosine_distance = total_cosine_distance / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - "
            + f"Val Loss: {avg_val_loss:.4f}"
            + f" - L2 Distance: {avg_l2_distance:.4f}"
            + f" - Cosine Distance: {avg_cosine_distance:.4f}"
        )

        if logging:
            wandb.log(
                {
                    "val/loss": avg_val_loss,
                    "l2_distance": avg_l2_distance,
                    "cosine_distance": avg_cosine_distance,
                }
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= task_config["training"]["patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    save_path = Path("ckpt") / "EQUI" / "FEAT" / transform / dataset / f"{feature}.pt"
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
    """Evaluate a model to predict the transformed feature given
    an original feature and a transformation parameter. Does
    feature extraction on-the-fly.

    Args:
        model: Model to evaluate.
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        transform: Name of the transform.
        label: Name of the label (factor of variation).
        task: Name of the downstream task.
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

    if isinstance(model, str) and logging:
        # Load model from wandb artifact
        model_wandb_path = f"{entity}/{project}/{model_name}"
        artifact_name = (
            f"{model_wandb_path}:latest" if ":" not in model_name else model_name
        )
        artifact = wandb.Api().artifact(f"{entity}/{project}/{artifact_name}")
        artifact_dir = artifact.download()
        model = get_probe(
            model_type=task_config["model"]["type"],
            in_features=feature_config["feature_dim"],
            emb_param=True,
            n_outputs=feature_config["feature_dim"],
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
    if transform == "PitchShift" or transform == "AddWhiteNoise":
        transform_obj = get_transform(
            transform_config,
            sample_rate=feature_config["sample_rate"],
        )
    elif dataset == "ImageNet":
        transform_obj = None
    else:
        transform_obj = get_transform(transform_config)

    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=task_config["evaluation"]["batch_size"]
    )

    model.eval()
    total_loss = 0
    total_l2_distance = 0
    total_cosine_distance = 0

    criterion = task_configs[task]["training"]["criterion"]()

    with torch.no_grad():
        for batch_raw_data, batch_targets in tqdm(test_loader, desc="Evaluating"):
            # prepare data for equivariance training
            original_features, transformed_features, transform_params = (
                preprocess_batch(
                    batch_raw_data=batch_raw_data,
                    batch_targets=batch_targets,
                    transform_obj=transform_obj,
                    transform=transform,
                    feature_extractor=feature_extractor,
                    sample_rate=feature_config.get("sample_rate", None),
                    device=device,
                )
            )

            # add parameter to original features - currently both are (b, c, 1)
            # concat_features = torch.cat([original_features, transform_params], dim=2)
            if transform_params.dim() == 3:
                transform_params = transform_params.squeeze(1)
            predicted_features = model(original_features, param=transform_params)
            if predicted_features.dim() == 2 and original_features.dim() == 3:
                predicted_features = predicted_features.unsqueeze(1)
            loss = criterion(original_features, predicted_features)

            total_loss += loss.item()

            # Compute L2 distance
            l2_distance = F.mse_loss(
                predicted_features, transformed_features, reduction="sum"
            )
            total_l2_distance += l2_distance.item()

            # Compute cosine distance
            cosine_distance = (
                1
                - F.cosine_similarity(predicted_features, transformed_features, dim=1)
                .sum()
                .item()
            )
            total_cosine_distance += cosine_distance

    avg_loss = total_loss / len(test_loader)
    avg_l2_distance = total_l2_distance / len(test_loader.dataset)
    avg_cosine_distance = total_cosine_distance / len(test_loader.dataset)

    print(f"Average test loss: {avg_loss:.4f}")
    print(f"Average L2 distance: {avg_l2_distance:.4f}")
    print(f"Average cosine distance: {avg_cosine_distance:.4f}")

    if logging:
        # Create a table for the evaluation metrics
        metrics_table = wandb.Table(columns=["Metric", "Value"])
        metrics_table.add_data("Average Loss", avg_loss)
        metrics_table.add_data("Average L2 Distance", avg_l2_distance)
        metrics_table.add_data("Average Cosine Distance", avg_cosine_distance)

        wandb.log({"evaluation_metrics": metrics_table})
        wandb.finish()

    return {
        "avg_loss": avg_loss,
        "avg_l2_distance": avg_l2_distance,
        "avg_cosine_distance": avg_cosine_distance,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model to predict transformed features."
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
