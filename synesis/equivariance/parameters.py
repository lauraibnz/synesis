"""Methods for training and evaluating a model to predict the
transformation parameter of a given original and augmented
feature pair.
"""

import argparse
import math
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
from synesis.utils import deep_update, get_artifact, get_wandb_config


def preprocess_batch(
    batch_raw_data,
    batch_targets,
    transform_obj,
    transform,
    feature_extractor,
    sample_rate,
    device,
    task_type="regressor",
    transform_config=None,
):
    """Get transformed data, extract features from both the original and
    transformed data, and concatenate them for input to the model."""

    if any(
        tf in transform
        for tf in ["HueShift", "BrightnessShift", "SaturationShift", "JPEGCompression", "InstrumentShift"]
    ):
        original_raw_data = batch_raw_data[:, 0].to(device)
        transformed_raw_data = batch_raw_data[:, 1].to(device)
        transform_params = batch_targets        
        if len(transform_params.shape) == 1:
            transform_params = transform_params.unsqueeze(1)
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
            transform_param = (math.log2(transform_param) + 1) / 2
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
            # Convert ratios to semitones: semitones = 12 * log2(ratio)
            # Then map [min_semitones, max_semitones] to [0, 1] for perceptually linear progression
            
            # Get the actual min/max from the passed transform config
            min_semitones = transform_config.get("params", {}).get("min_transpose_semitones", -12)
            max_semitones = transform_config.get("params", {}).get("max_transpose_semitones", 12)
            
            # Convert ratios to semitones, then normalize to [0, 1]
            semitone_params = [12 * math.log2(ratio) for ratio in transform_params]
            transform_params = [(s - min_semitones) / (max_semitones - min_semitones) for s in semitone_params]

            # convert to tensor of shape [batch, 1, 1] and move to device
            transform_params = torch.tensor(transform_params).to(device)
        elif "AddWhiteNoise" in transform:
            transform_params = transform_obj.transform_parameters["snr_in_db"]
            # map [-30, 50] to [1, 0]
            transform_params = 1 - (transform_params + 30) / 80
            # convert to tensor of shape [batch, 1, 1] and move to device
            transform_params = torch.tensor(transform_params).to(device)
        elif "LowPassFilter" in transform:
            transform_params = transform_obj.transform_parameters["cutoff_freq"]
            # Get the actual min/max from the passed transform config
            min_cutoff = transform_config.get("params", {}).get("min_cutoff_freq", 1000)
            max_cutoff = transform_config.get("params", {}).get("max_cutoff_freq", 4000)
            
            # Normalize [min_cutoff, max_cutoff] to [0, 1]
            transform_params = [(freq - min_cutoff) / (max_cutoff - min_cutoff) for freq in transform_params]
            
            # convert to tensor of shape [batch, 1, 1] and move to device
            transform_params = torch.tensor(transform_params).to(device)
        else:
            # they will be of shape [batch, channel, 1], and on device
            transform_params = transform_obj.transform_parameters[
                f"{transform.lower()}_factors"
            ]

    # For classification tasks, we need to handle discrete class labels
    if task_type == "classifier":
        # For classification, batch_targets should already be class indices
        # We don't need to process them further, just ensure they're on the right device
        transform_params = batch_targets.to(device)

    # combine original and transformed data
    combined_raw_data = torch.cat([original_raw_data, transformed_raw_data], dim=0)

    with torch.no_grad():
        combined_features = feature_extractor(combined_raw_data)
        if combined_features.dim() > 1:
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
        wandb_config = get_wandb_config()
        run_name = f"EQUI_PARA_{transform}_{label}_{dataset}_{feature}"
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
        transform=transform,
        label=label,
        split="train",
        download=False,
        item_format="raw",
        itemization=False,
    )
    val_dataset = get_dataset(
        name=dataset,
        feature=feature,
        transform=transform,
        label=label,
        split="validation",
        download=False,
        item_format="raw",
        itemization=False,
    )

    # If dataset returns subitems per item, need to wrap it
    if transform_config and train_dataset[0][0].dim() == 3:
        wrapped_train = SubitemDataset(train_dataset)
        wrapped_val = SubitemDataset(val_dataset)
        del train_dataset, val_dataset
        train_dataset = wrapped_train
        val_dataset = wrapped_val

    feature_extractor = get_feature_extractor(feature)
    feature_extractor = feature_extractor.to(device)
    if any(tf in transform for tf in ["PitchShift", "AddWhiteNoise", "LowPassFilter"]):
        transform_obj = get_transform(
            transform_config,
            sample_rate=feature_config["sample_rate"],
        )
    elif not transform_config:
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

    sample_item, sample_target = train_dataset[0]
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

    try:
        n_outputs = sample_target.shape[0]
    except:
        n_outputs = 1

    # For classification tasks, n_outputs should be the number of classes
    if task_config["model"]["type"] == "classifier":
        # Special case: if target is multi-dimensional (like one-hot difference vectors),
        # use the target dimension as n_outputs
        if len(sample_target.shape) > 1 or sample_target.shape[0] > 1:
            n_outputs = sample_target.shape[0]
        else:
            n_outputs = len(train_dataset.label_encoder.classes_)

    model = get_probe(
        model_type=task_config["model"]["type"],
        in_features=in_features * 2,
        n_outputs=n_outputs,
        use_temporal_pooling=use_temporal_pooling,
        **task_config["model"]["params"],
    ).to(device)

    criterion = task_config["training"]["criterion"]()
    optimizer_class = task_config["training"]["optimizer"]["class"]
    optimizer = optimizer_class(
        model.parameters(),
        **task_config["training"]["optimizer"]["params"],
    )

    # For multi-dimensional targets, use MSE loss regardless of config
    if task_config["model"]["type"] == "classifier" and len(sample_target.shape) > 1:
        criterion = nn.MSELoss()

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
                task_type=task_config["model"]["type"],
                transform_config=transform_config,
            )

            optimizer.zero_grad()
            predicted_params = model(concat_features)
            # Only squeeze for regression tasks, not for classification
            if task_config["model"]["type"] == "regressor" and len(predicted_params.shape) == 2:
                predicted_params = predicted_params.squeeze(1)
            
            # For classification, targets should be long tensors
            if task_config["model"]["type"] == "classifier":
                # Only convert to long if it's a single class index
                if len(transform_params.shape) == 1 and transform_params.shape[0] == 1:
                    transform_params = transform_params.long()
            
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
                    task_type=task_config["model"]["type"],
                    transform_config=transform_config,
                )

                predicted_params = model(concat_features)
                # Only squeeze for regression tasks, not for classification
                if task_config["model"]["type"] == "regressor" and len(predicted_params.shape) == 2:
                    predicted_params = predicted_params.squeeze(1)
                
                # For classification, targets should be long tensors
                if task_config["model"]["type"] == "classifier":
                    # Only convert to long if it's a single class index
                    if len(transform_params.shape) == 1 and transform_params.shape[0] == 1:
                        transform_params = transform_params.long()
                
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
        itemization=False,
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
    if transform_config and test_dataset[0][0].dim() == 3:
        wrapped_test = SubitemDataset(test_dataset)
        del test_dataset
        test_dataset = wrapped_test

    feature_extractor = get_feature_extractor(feature)
    feature_extractor = feature_extractor.to(device)
    if any(tf in transform for tf in ["PitchShift", "AddWhiteNoise", "LowPassFilter"]):
        transform_obj = get_transform(
            transform_config,
            sample_rate=feature_config["sample_rate"],
        )
    elif not transform_config:
        # transform handled in dataset
        transform_obj = None
    else:
        transform_obj = get_transform(transform_config)

    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=task_config["evaluation"]["batch_size"]
    )

    sample_item, sample_target = test_dataset[0]

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

    try:
        n_outputs = sample_target.shape[0]
    except:
        n_outputs = 1    

    # For classification tasks, n_outputs should be the number of classes
    if task_config["model"]["type"] == "classifier":
        # Special case: if target is multi-dimensional (like one-hot difference vectors),
        # use the target dimension as n_outputs
        if len(sample_target.shape) > 1 or sample_target.shape[0] > 1:
            n_outputs = sample_target.shape[0]
        else:
            n_outputs = len(test_dataset.label_encoder.classes_)

    if isinstance(model, str):
        # Load model from wandb artifact
        artifact = get_artifact(model)
        artifact_dir = artifact.download()

    model = get_probe(
        model_type=task_config["model"]["type"],
        in_features=in_features * 2,
        n_outputs=n_outputs,
        use_temporal_pooling=use_temporal_pooling,
        **task_config["model"]["params"],
    )
    model.load_state_dict(torch.load(Path(artifact_dir) / f"{feature}.pt"))
    os.remove(Path(artifact_dir) / f"{feature}.pt")

    model.to(device)

    model.eval()
    total_loss = 0
    test_metric_results = {}
    criterion = task_config["training"]["criterion"]()

    # For multi-dimensional targets, use MSE loss regardless of config
    if task_config["model"]["type"] == "classifier" and len(sample_target.shape) > 1:
        criterion = nn.MSELoss()

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
                task_type=task_config["model"]["type"],
                transform_config=transform_config,
            )

            predicted_params = model(concat_features)
            # Only squeeze for regression tasks, not for classification
            if task_config["model"]["type"] == "regressor" and len(predicted_params.shape) == 2:
                predicted_params = predicted_params.squeeze(1)
            
            # For classification, targets should be long tensors
            if task_config["model"]["type"] == "classifier":
                # Only convert to long if it's a single class index
                if len(transform_params.shape) == 1 and transform_params.shape[0] == 1:
                    transform_params = transform_params.long()
            
            loss = criterion(predicted_params, transform_params)

            total_loss += loss.item()

            # Calculate appropriate metrics based on task type
            if task_config["model"]["type"] == "classifier":
                # Check if this is a multi-dimensional target (like one-hot difference vectors)
                if len(transform_params.shape) > 1 or transform_params.shape[0] > 1:
                    # For vector predictions, use cosine similarity
                    cos_sim = nn.CosineSimilarity(dim=-1)(predicted_params, transform_params).mean().item()
                    if "cosine_similarity" not in test_metric_results:
                        test_metric_results["cosine_similarity"] = 0
                    test_metric_results["cosine_similarity"] += cos_sim
                else:
                    # For classification, calculate accuracy
                    if predicted_params.dim() > 1:
                        predicted_classes = predicted_params.argmax(dim=-1)
                    else:
                        predicted_classes = predicted_params
                    # Ensure both tensors have the same shape for comparison
                    if predicted_classes.shape != transform_params.shape:
                        predicted_classes = predicted_classes.squeeze()
                        transform_params = transform_params.squeeze()
                    accuracy = (predicted_classes == transform_params).float().mean().item()
                    if "accuracy" not in test_metric_results:
                        test_metric_results["accuracy"] = 0
                    test_metric_results["accuracy"] += accuracy
            else:
                # For regression, calculate MSE and MAE
                mse = nn.MSELoss()(predicted_params, transform_params).item()
                if "mse" not in test_metric_results:
                    test_metric_results["mse"] = 0
                test_metric_results["mse"] += mse

                mae = nn.L1Loss()(predicted_params, transform_params).item()
                if "mae" not in test_metric_results:
                    test_metric_results["mae"] = 0
                test_metric_results["mae"] += mae

    # Average metrics
    avg_loss = total_loss / len(test_loader)
    
    if task_config["model"]["type"] == "classifier":
        # Check if we have vector predictions or class predictions
        if "cosine_similarity" in test_metric_results:
            # Vector predictions (like SynTheory InstrumentShift)
            cosine_similarity = test_metric_results["cosine_similarity"] / len(test_loader)
            print(f"Average test loss: {avg_loss:.4f}")
            print(f"Cosine Similarity: {cosine_similarity:.4f}")
            
            if logging:
                # Log vector prediction metrics
                wandb.log({
                    "test/CosineSimilarity": cosine_similarity,
                })
                
                # Create a table for the evaluation metrics
                metrics_table = wandb.Table(columns=["Metric", "Value"])
                metrics_table.add_data("Average Loss", avg_loss)
                metrics_table.add_data("Cosine Similarity", cosine_similarity)
        else:
            # Class predictions
            accuracy = test_metric_results["accuracy"] / len(test_loader)
            print(f"Average test loss: {avg_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            
            if logging:
                # Log classification metrics
                wandb.log({
                    "test/Accuracy": accuracy,
                })
                
                # Create a table for the evaluation metrics
                metrics_table = wandb.Table(columns=["Metric", "Value"])
                metrics_table.add_data("Average Loss", avg_loss)
                metrics_table.add_data("Accuracy", accuracy)
    else:
        mse = test_metric_results["mse"] / len(test_loader)
        mae = test_metric_results["mae"] / len(test_loader)
        print(f"Average test loss: {avg_loss:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        if logging:
            # Log regression metrics
            wandb.log({
                "test/MSE": mse,
                "test/MAE": mae,
            })
            
            # Create a table for the evaluation metrics
            metrics_table = wandb.Table(columns=["Metric", "Value"])
            metrics_table.add_data("Average Loss", avg_loss)
            metrics_table.add_data("Mean Squared Error", mse)
            metrics_table.add_data("Mean Absolute Error", mae)

    if logging:
        # Log the table to wandb
        wandb.log({"evaluation_metrics": metrics_table})
        wandb.finish()


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

    evaluate(
        model=model,
        feature=args.feature,
        dataset=args.dataset,
        transform=args.transform,
        label=args.label,
        task=args.task,
        device=args.device,
        logging=not args.nolog,
    )
