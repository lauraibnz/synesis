"""Methods for training and evaluating a model to predict the
transformed feature given an original feature and
a transformation parameter.
"""

import argparse
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.equivariance.features import configs as task_configs
from config.features import configs as feature_configs
from config.transforms import configs as transform_configs
from synesis.datasets.dataset_utils import get_dataset
from synesis.features.feature_utils import (
    DynamicBatchSampler,
    collate_packed_batch,
    get_feature_extractor,
)
from synesis.transforms.transform_utils import get_transform
from synesis.utils import deep_update


class TransformedPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, 256),  # +1 for the transform parameter
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x, param):
        combined = torch.cat([x, param.unsqueeze(1)], dim=1)
        return self.model(combined)


def train(
    feature: str,
    dataset: str,
    transform: str,
    task: str,
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
):
    """Train a model to predict the transformed feature given
    an original feature and a transformation parameter. Does
    feature extraction on-the-fly.

    Args:
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        transform: Name of the transform (factor of variation).
        task: Name of the downstream task.
        task_config: Override certain values of the task configuration.
        device: Device to use for training (defaults to "cuda" if available).
    """
    feature_config = feature_configs.get(feature)
    transform_config = transform_configs.get(transform)
    task_config = task_configs.get("default")
    if task in task_configs:
        task_config = deep_update(task_config, task_configs[task])

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
        name=dataset, feature=feature, split="val", download=False, item_format="raw"
    )

    assert (
        transform in train_dataset.transforms
    ), f"Transform {transform} not found in dataset {dataset}"

    feature_extractor = get_feature_extractor(feature)
    feature_extractor = feature_extractor.to(device)
    transform_obj = get_transform(transform_config)

    train_sampler = DynamicBatchSampler(
        dataset=train_dataset, batch_size=task_config["training"]["batch_size"]
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, collate_fn=collate_packed_batch
    )

    val_sampler = DynamicBatchSampler(
        dataset=val_dataset, batch_size=task_config["training"]["batch_size"]
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler, collate_fn=collate_packed_batch
    )

    model = TransformedPredictor(
        input_dim=len(train_dataset[0][0][0]), output_dim=len(train_dataset[0][0][0])
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=task_config["training"]["lr"])

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    num_epochs = task_config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_raw_data, _ in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            batch_raw_data = batch_raw_data.to(device)

            with torch.no_grad():
                original_features = feature_extractor(batch_raw_data)

            transformed_raw_data, transform_params = zip(
                *[transform_obj(raw_data) for raw_data in batch_raw_data]
            )
            transformed_raw_data = torch.stack(transformed_raw_data).to(device)
            transform_params = torch.tensor(transform_params).to(device)

            with torch.no_grad():
                transformed_features = feature_extractor(transformed_raw_data)

            optimizer.zero_grad()
            preadicted_features = model(original_features, transform_params)
            loss = criterion(preadicted_features, transformed_features)

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

                original_features = feature_extractor(batch_raw_data)

                transformed_raw_data, transform_params = zip(
                    *[transform_obj(raw_data) for raw_data in batch_raw_data]
                )
                transformed_raw_data = torch.stack(transformed_raw_data).to(device)
                transform_params = torch.tensor(transform_params).to(device)

                transformed_features = feature_extractor(transformed_raw_data)

                predicted_features = model(original_features, transform_params)
                loss = criterion(predicted_features, transformed_features)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - "
            + f"Val Loss: {avg_val_loss:.4f}"
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

    return model


def evaluate(
    model: nn.Module,
    feature: str,
    dataset: str,
    transform: str,
    task: str,
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
):
    feature_config = feature_configs.get(feature)
    transform_config = transform_configs.get(transform)
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = get_dataset(
        name=dataset, feature=feature, split="test", download=False, item_format="raw"
    )

    assert (
        transform in test_dataset.transforms
    ), f"Transform {transform} not available in {dataset}"

    feature_extractor = get_feature_extractor(feature)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    transform_obj = get_transform(transform_config)

    test_sampler = DynamicBatchSampler(
        dataset=test_dataset, batch_size=task_config["evaluation"]["batch_size"]
    )
    test_loader = DataLoader(
        test_dataset, batch_sampler=test_sampler, collate_fn=collate_packed_batch
    )

    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_raw_data, _ in tqdm(test_loader, desc="Evaluating"):
            batch_raw_data = batch_raw_data.to(device)

            original_features = feature_extractor(batch_raw_data)

            transformed_raw_data, transform_params = transform_obj(batch_raw_data)

            transformed_features = feature_extractor(transformed_raw_data)

            predicted_features = model(original_features, transform_params)
            loss = criterion(predicted_features, transformed_features)

            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Average test loss: {avg_loss:.4f}")

    return {"avg_loss": avg_loss}


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
