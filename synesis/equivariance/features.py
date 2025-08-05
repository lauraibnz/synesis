"""Methods for training and evaluating a model to predict the
transformed feature given an original feature and
a transformation parameter.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
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
from synesis.utils import deep_update, get_artifact, get_wandb_config


def load_feature_stats(feature: str, dataset_name: str):
    """Load pre-computed mean and std for feature normalization."""
    # Construct stats filename
    stats_filename = f"mean_std_{dataset_name}_{feature}.txt"
    
    stats_path = Path("stats") / stats_filename
    with open(stats_path) as f:
        lines = f.readlines()
        mean = float(lines[0].split(": ")[1])
        std = float(lines[1].split(": ")[1])
    return mean, std


def flatten_features(tensor):
    """Flatten features for consistent processing."""
    # If shape is [B, 1, C, T], do temporal mean pooling
    if tensor.dim() == 4:
        tensor = tensor.mean(dim=-1)  # Pool over time -> [B, 1, C]
    return tensor


def preprocess_batch(
    batch_raw_data,
    batch_targets,
    transform_obj,
    transform,
    feature_extractor,
    sample_rate,
    device,
):
    if any(
        tf in transform
        for tf in ["HueShift", "BrightnessShift", "SaturationShift", "JPEGCompression"]
    ):
        original_raw_data = batch_raw_data[:, 0].to(device)
        transformed_raw_data = batch_raw_data[:, 1].to(device)
        transform_params = batch_targets
        if len(transform_params.shape) == 1:
            transform_params = transform_params.unsqueeze(1)
        if len(transform_params.shape) == 2:
            transform_params = transform_params.unsqueeze(2)
        transform_params = transform_params.to(device)

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
        if "PitchShift" in transform:
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
        elif "AddWhiteNoise" in transform:
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
        if combined_features.dim() > 1:
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
    debug: bool = False,
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

    if logging or debug:
        run_name = f"2_EQUI_FEAT_{task}_{transform}_{label}_{dataset}_{feature}"
        if debug:
            run_name = f"DEBUG_{run_name}"
        wandb_config = get_wandb_config()
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
    feature_mean, feature_std = load_feature_stats(feature, dataset)

    if any(tf in transform for tf in ["PitchShift", "AddWhiteNoise"]):
        transform_obj = get_transform(
            transform_config,
            sample_rate=feature_config["sample_rate"],
        )
    elif dataset == "ImageNet":
        transform_obj = None
    else:
        transform_obj = get_transform(transform_config)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=task_config["training"]["batch_size"],
        drop_last=True if task_config["model"]["batch_norm"] else False,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=task_config["training"]["batch_size"],
        drop_last=True if task_config["model"]["batch_norm"] else False,
    )

    sample_item, _ = train_dataset[0]
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

    model = get_probe(
        model_type=task_config["model"]["type"],
        in_features=in_features,
        emb_param=task_config["model"]["emb_param"],
        emb_param_dim=task_config["model"]["emb_param_dim"],
        n_outputs=in_features,
        use_batch_norm=task_config["model"]["batch_norm"],
        use_temporal_pooling=use_temporal_pooling,
        **task_config["model"]["params"],
    ).to(device)

    criterion = task_config["training"]["criterion"]()
    optimizer_class = task_config["training"]["optimizer"]["class"]
    optimizer = optimizer_class(
        model.parameters(),
        **task_config["training"]["optimizer"]["params"],
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    num_epochs = task_config["training"]["num_epochs"]
    mini_val_interval = 0.05
    mini_val_step = int(len(train_loader) * mini_val_interval)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (batch_raw_data, batch_targets) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
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
            if transform_params.dim() > 2:
                transform_params = transform_params.squeeze(1)

            if task_config["model"]["feature_norm"] or feature == "XVector":
                original_features = (original_features - feature_mean) / feature_std
                transformed_features = (
                    transformed_features - feature_mean
                ) / feature_std
            if task_config["model"]["batch_norm"]:
                original_features = model.input_batch_norm(original_features.squeeze(1))
                transformed_features = model.input_batch_norm(
                    transformed_features.squeeze(1)
                )

            predicted_features = model(original_features, param=transform_params)
            if original_features.dim() > predicted_features.dim():
                predicted_features = predicted_features.unsqueeze(1)

            if use_temporal_pooling:
                transformed_features = flatten_features(transformed_features)

            loss = criterion(predicted_features, transformed_features)

            loss.backward()

            if debug:
                # Log gradient stats to wandb
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log(
                            {
                                f"grad/{name}.mean": param.grad.mean().item(),
                                f"grad/{name}.std": param.grad.std().item(),
                            }
                        )
                # Log plot with original, transformed, and predicted features
                for i in range(original_features.shape[0]):
                    fig, ax = plt.subplots()
                    ax.plot(
                        original_features[i][0].detach().cpu().numpy(), label="Original"
                    )
                    ax.plot(
                        transformed_features[i][0].detach().cpu().numpy(),
                        label="Transformed",
                    )
                    ax.plot(
                        predicted_features[i][0].detach().cpu().numpy(),
                        label="Predicted",
                    )
                    ax.set_title(f"Transform Param: {transform_params[i].item():.4f}")
                    ax.legend()
                    wandb.log({f"feature_plots/plot_{i}": wandb.Image(fig)})
                    plt.close(fig)

            optimizer.step()

            total_train_loss += loss.item()

            if logging or debug:
                wandb.log({"train/loss": loss.item()})

            # Mini validation step every x% of an epoch, if specified
            if mini_val_step and (batch_idx + 1) % mini_val_step == 0:
                model.eval()
                mini_val_loss = 0
                mini_val_cosine_similarity = 0
                num_mini_val_items = 0
                with torch.no_grad():
                    for val_batch_raw_data, val_batch_targets in val_loader:
                        (
                            val_original_features,
                            val_transformed_features,
                            val_transform_params,
                        ) = preprocess_batch(
                            batch_raw_data=val_batch_raw_data,
                            batch_targets=val_batch_targets,
                            transform_obj=transform_obj,
                            transform=transform,
                            feature_extractor=feature_extractor,
                            sample_rate=feature_config.get("sample_rate", None),
                            device=device,
                        )

                        if val_transform_params.dim() == 3:
                            val_transform_params = val_transform_params.squeeze(1)

                        if task_config["model"]["feature_norm"] or feature == "XVector":
                            val_original_features = (
                                val_original_features - feature_mean
                            ) / feature_std
                            val_transformed_features = (
                                val_transformed_features - feature_mean
                            ) / feature_std
                        if model.use_batch_norm:
                            val_original_features = model.input_batch_norm(
                                val_original_features.squeeze(1)
                            )
                            val_transformed_features = model.input_batch_norm(
                                val_transformed_features.squeeze(1)
                            )

                        val_predicted_features = model(
                            val_original_features, param=val_transform_params
                        )
                        if ( 
                            val_original_features.dim() > val_predicted_features.dim()
                        ):
                            val_predicted_features = val_predicted_features.unsqueeze(1)

                        if use_temporal_pooling:
                            val_transformed_features = flatten_features(val_transformed_features)

                        val_loss = criterion(
                            val_predicted_features, val_transformed_features
                        )
                        mini_val_loss += val_loss.item()

                        # Compute cosine similarity
                        cosine_similarity = (
                            F.cosine_similarity(
                                val_predicted_features, val_transformed_features, dim=1
                            )
                            .mean()
                            .item()
                        )
                        mini_val_cosine_similarity += cosine_similarity

                        num_mini_val_items += val_batch_raw_data.size(0)
                        if num_mini_val_items >= 500:
                            break

                avg_mini_val_loss = mini_val_loss / (
                    num_mini_val_items / val_loader.batch_size
                )
                avg_mini_val_cosine_similarity = mini_val_cosine_similarity / (
                    num_mini_val_items / val_loader.batch_size
                )
                if logging or debug:
                    wandb.log(
                        {
                            "mini_val/loss": avg_mini_val_loss,
                            "mini_val/cosine_similarity": avg_mini_val_cosine_similarity,
                        }
                    )
                model.train()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        total_l2_distance = 0
        total_cosine_similarity = 0
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

                if task_config["model"]["feature_norm"] or feature == "XVector":
                    original_features = (original_features - feature_mean) / feature_std
                    transformed_features = (
                        transformed_features - feature_mean
                    ) / feature_std
                if model.use_batch_norm:
                    original_features = model.input_batch_norm(
                        original_features.squeeze(1)
                    )
                    transformed_features = model.input_batch_norm(
                        transformed_features.squeeze(1)
                    )

                predicted_features = model(original_features, param=transform_params)
                if original_features.dim() > predicted_features.dim():
                    predicted_features = predicted_features.unsqueeze(1)

                if use_temporal_pooling:
                    transformed_features = flatten_features(transformed_features)

                loss = criterion(predicted_features, transformed_features)

                total_val_loss += loss.item()

                # Compute L2 distance
                l2_distance = F.mse_loss(
                    predicted_features, transformed_features, reduction="mean"
                )
                total_l2_distance += l2_distance.item()

                # Compute cosine distance
                cosine_similarity = (
                    F.cosine_similarity(predicted_features, transformed_features, dim=1)
                    .mean()
                    .item()
                )
                total_cosine_similarity += cosine_similarity

            avg_val_loss = total_val_loss / len(val_loader)
            avg_l2_distance = total_l2_distance / len(val_loader)
            avg_cosine_similarity = total_cosine_similarity / len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - "
            + f"Val Loss: {avg_val_loss:.4f}"
            + f" - L2 Distance: {avg_l2_distance:.4f}"
            + f" - Cosine Distance: {avg_cosine_similarity:.4f}"
        )

        if logging or debug:
            wandb.log(
                {
                    "val/loss": avg_val_loss,
                    "l2_distance": avg_l2_distance,
                    "cosine_similarity": avg_cosine_similarity,
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
    if logging or debug:
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
    debug: bool = False,
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
        if logging or debug:
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
    feature_mean, feature_std = load_feature_stats(feature, dataset)

    if any(tf in transform for tf in ["PitchShift", "AddWhiteNoise"]):
        transform_obj = get_transform(
            transform_config,
            sample_rate=feature_config["sample_rate"],
        )
    elif dataset == "ImageNet":
        transform_obj = None
    else:
        transform_obj = get_transform(transform_config)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=task_config["evaluation"]["batch_size"],
        drop_last=True if task_config["model"]["batch_norm"] else False,
    )

    sample_item, _ = test_dataset[0]
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
        model = get_probe(
            model_type=task_config["model"]["type"],
            in_features=in_features,
            emb_param=task_config["model"]["emb_param"],
            emb_param_dim=task_config["model"]["emb_param_dim"],
            n_outputs=in_features,
            use_batch_norm=task_config["model"]["batch_norm"],
            use_temporal_pooling=use_temporal_pooling,
            **task_config["model"]["params"],
        ).to(device)
        model.load_state_dict(torch.load(Path(artifact_dir) / f"{feature}.pt"))
        os.remove(Path(artifact_dir) / f"{feature}.pt")

    model.to(device)

    model.eval()
    total_loss = 0
    total_l2_distance = 0
    total_cosine_similarity = 0

    criterion = task_config["training"]["criterion"]()

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

            if task_config["model"]["feature_norm"] or feature == "XVector":
                original_features = (original_features - feature_mean) / feature_std
                transformed_features = (
                    transformed_features - feature_mean
                ) / feature_std
            if task_config["model"]["batch_norm"]:
                original_features = model.input_batch_norm(original_features.squeeze(1))
                transformed_features = model.input_batch_norm(
                    transformed_features.squeeze(1)
                )

            predicted_features = model(original_features, param=transform_params)
            if original_features.dim() > predicted_features.dim():
                predicted_features = predicted_features.unsqueeze(1)

            if use_temporal_pooling:
                transformed_features = flatten_features(transformed_features)

            loss = criterion(predicted_features, transformed_features)

            total_loss += loss.item()

            # Compute L2 distance
            l2_distance = F.mse_loss(
                predicted_features, transformed_features, reduction="mean"
            )
            total_l2_distance += l2_distance.item()

            # Compute cosine distance
            cosine_similarity = (
                F.cosine_similarity(predicted_features, transformed_features, dim=1)
                .mean()
                .item()
            )
            total_cosine_similarity += cosine_similarity

    avg_loss = total_loss / len(test_loader)
    avg_l2_distance = total_l2_distance / len(test_loader)
    avg_cosine_similarity = total_cosine_similarity / len(test_loader)

    print(f"Average test loss: {avg_loss:.4f}")
    print(f"Average L2 distance: {avg_l2_distance:.4f}")
    print(f"Average cosine distance: {avg_cosine_similarity:.4f}")

    if logging or debug:
        # Create a table for the evaluation metrics
        metrics_table = wandb.Table(columns=["Metric", "Value"])
        metrics_table.add_data("Average Loss", avg_loss)
        metrics_table.add_data("Average L2 Distance", avg_l2_distance)
        metrics_table.add_data("Average Cosine Similarity", avg_cosine_similarity)

        wandb.log({"evaluation_metrics": metrics_table})
        wandb.finish()

    return {
        "avg_loss": avg_loss,
        "avg_l2_distance": avg_l2_distance,
        "avg_cosine_similarity": avg_cosine_similarity,
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
    parser.add_argument(
        "--debug",
        action="store_true",
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
        debug=args.debug,
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
        debug=args.debug,
    )
