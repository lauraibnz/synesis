"""Methods for training and evaluating a model to predict the
transformation parameter of a given original and augmented
representation pair.
"""

import argparse
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.transforms import transform_configs
from synesis.datasets.dataset_utils import get_dataset
from synesis.features.feature_utils import (
    DynamicBatchSampler,
    collate_packed_batch,
    get_feature_extractor,
)
from synesis.transforms.transform_utils import get_transform
from synesis.utils import deep_update


def train(
    feature: str,
    dataset: str,
    transform: str,
    transform_config: Optional[dict] = None,
    device: Optional[str] = None,
    num_epochs=50,
    batch_size=32,
    learning_rate=0.001,
    patience=10,
):
    """Train a model to predict the transformation parameter of
    a given original and augmented representation pair. Does
    feature extraction on-the-fly.

    Args:
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        transform: Name of the transform (factor of variation).
        transform_config: Override certain values of the transform config.
        device: Device to use for training (defaults to "cuda" if available).
    """

    if transform_config:
        transform_configs[transform] = deep_update(
            transform_configs[transform], transform_config
        )

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="train",
        download=False,
        item_format="raw",
    )
    val_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="validation",
        download=False,
        item_format="raw",
    )

    assert (
        transform in train_dataset.transforms
    ), f"Transform {transform} not available in {dataset}"

    feature_extractor, extract_kws = get_feature_extractor(feature)
    feature_extractor = feature_extractor.to(device)

    transform_obj = get_transform(transform_configs[transform])

    train_sampler = DynamicBatchSampler(dataset=train_dataset, batch_size=batch_size)
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, collate_fn=collate_packed_batch
    )

    val_sampler = DynamicBatchSampler(dataset=val_dataset, batch_size=batch_size)
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler, collate_fn=collate_packed_batch
    )

    model = nn.Sequential(
        nn.Linear(len(train_dataset[0][0][0]) * 2, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch_raw_data, _ in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            batch_raw_data = batch_raw_data.to(device)

            with torch.no_grad():
                original_features = feature_extractor(batch_raw_data, **extract_kws)

            transformed_raw_data, transform_params = zip(
                *[transform_obj(raw_data) for raw_data in batch_raw_data]
            )
            transformed_raw_data = torch.stack(transformed_raw_data).to(device)
            transform_params = torch.tensor(transform_params).float().to(device)

            with torch.no_grad():
                transformed_features = feature_extractor(transformed_raw_data, **extract_kws)

            combined_features = torch.cat(
                [original_features, transformed_features], dim=1
            )

            optimizer.zero_grad()
            predicted_params = model(combined_features)
            loss = criterion(predicted_params, transform_params)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_raw_data, _ in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                batch_raw_data = batch_raw_data.to(device)

                original_features = feature_extractor(batch_raw_data, **extract_kws)

                transformed_raw_data, transform_params = zip(
                    *[transform_obj(raw_data) for raw_data in batch_raw_data]
                )
                transformed_raw_data = torch.stack(transformed_raw_data).to(device)
                transform_params = torch.tensor(transform_params).float().to(device)

                transformed_features = feature_extractor(transformed_raw_data, **extract_kws)

                combined_features = torch.cat(
                    [original_features, transformed_features], dim=1
                )

                predicted_params = model(combined_features)
                loss = criterion(predicted_params, transform_params)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
            + f"Val Loss: {avg_val_loss:.4f}"
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

    return model


def evaluate(
    model: nn.Module,
    feature: str,
    dataset: str,
    transform: str,
    transform_config: Optional[dict] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
):
    """
    Evaluate a given trained model for predicting transformation parameters.

    Args:
        model: Trained model for predicting transformation parameters.
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
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
        item_format="raw",
    )

    assert (
        transform in test_dataset.transforms
    ), f"Transform {transform} not available in {dataset}"

    feature_extractor, extract_kws = get_feature_extractor(feature)
    feature_extractor = feature_extractor.to(device)

    transform_obj = get_transform(transform_configs[transform])

    test_sampler = DynamicBatchSampler(dataset=test_dataset, batch_size=batch_size)
    test_loader = DataLoader(
        test_dataset, batch_sampler=test_sampler, collate_fn=collate_packed_batch
    )

    model.eval()
    total_loss = 0
    all_predicted_params = []
    all_true_params = []

    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_raw_data, _ in tqdm(test_loader, desc="Evaluating"):
            batch_raw_data = batch_raw_data.to(device)

            original_features = feature_extractor(batch_raw_data, **extract_kws)

            transformed_raw_data, transform_params = transform_obj(batch_raw_data)

            transformed_features = feature_extractor(transformed_raw_data, **extract_kws)

            combined_features = torch.cat(
                [original_features, transformed_features], dim=1
            )

            predicted_params = model(combined_features)
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

    return {"avg_loss": avg_loss, "mse": mse, "mae": mae, "r_squared": r_squared.item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transform prediction model.")
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
        "-t",
        type=str,
        required=True,
        help="Data transform name.",
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
        transform=args.transform,
        device=args.device,
    )

    results = evaluate(
        model=model,
        feature=args.feature,
        dataset=args.dataset,
        transform=args.transform,
        device=args.device,
    )
