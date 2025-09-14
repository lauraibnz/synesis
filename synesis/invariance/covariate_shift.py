import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config.features import configs as feature_configs
from config.transforms import configs as transform_configs
from synesis.datasets.dataset_utils import SubitemDataset, get_dataset
from synesis.features.feature_utils import get_feature_extractor
from synesis.transforms.transform_utils import get_transform
from synesis.utils import get_wandb_config, get_artifact


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
        for tf in ["HueShift", "BrightnessShift", "SaturationShift", "JPEGCompression", "InstrumentShift"]
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
        elif "LowPassFilter" in transform:
            transform_params = transform_obj.transform_parameters["cutoff_freq"]
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

    # currently, features are of shape (2b, c, t), where the first half of the
    # batch is originals, and the second is transformed. We need to split them
    # such that original feature 0 is concatenated with transformed 0, etc.
    original_features, transformed_features = torch.split(
        combined_features, batch_raw_data.size(0), dim=0
    )

    return original_features, transformed_features, transform_params

def flatten_features(tensor):
    # If shape is [B, 1, C, T], do temporal mean pooling
    if tensor.dim() == 4:
        tensor = tensor.mean(dim=-1)  # Pool over time -> [B, 1, C]
    # If shape is [B, 1, C], remove singleton dim -> [B, C]
    if tensor.dim() == 3:
        tensor = tensor.squeeze(1)
    return tensor

def feature_distances(
    feature: str,
    dataset: str,
    transform: str,
    batch_size: int = 32,
    passes: int = 3,
    device: Optional[str] = None,
    logging: bool = True,
    label: str = "dummy",
):
    """Calculate distances between features of original and transformed data.
    Meant to test robustness to covariate shifts.

    Args:
        feature: Name of the feature to calculate distances for.
        dataset: Name of the dataset to use.
        transform: Name of the transform to apply.
        item_format: Format of the items in the dataset.
        device: Device to use for the computation.
        logging: Whether to log the results.
    Returns:
        A dictionary containing mean metrics.
    """
    feature_config = feature_configs.get(feature)
    transform_config = transform_configs.get(transform)

    if logging:
        wandb_config = get_wandb_config()
        run_name = f"INVAR_{transform}_{dataset}_{feature}"
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=run_name,
            config={
                "feature": feature,
                "feature_config": feature_config,
                "dataset": dataset,
            },
        )
        artifact = wandb.Artifact(run_name, type="model")

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the dataset
    raw_dataset = get_dataset(
        name=dataset,
        feature=feature,
        transform=transform,
        label=label,
        split="test",
        download=False,
        item_format="raw",
        itemization=False,
    )

    # If dataset returns subitems per item, need to wrap it
    if transform_config and raw_dataset[0][0].dim() == 3:
        wrapped_dataset = SubitemDataset(raw_dataset)
        del raw_dataset
        raw_dataset = wrapped_dataset

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

    loader = DataLoader(raw_dataset, shuffle=False, batch_size=batch_size)

    total_l1 = []
    total_l2 = []
    total_cosine = []

    for pass_ in range(passes):
        with tqdm(total=len(loader), desc=f"Pass {pass_+1}/{passes}") as pbar:
            for items, labels in loader:
                # extract features
                original_features, transformed_features, transform_params = preprocess_batch(
                    items,
                    labels,
                    transform_obj,
                    transform,
                    feature_extractor,
                    feature_config.get("sample_rate", None),
                    device,
                )
                original_features = flatten_features(original_features)
                transformed_features = flatten_features(transformed_features)

                l1 = torch.norm(original_features - transformed_features, p=1, dim=1)
                l2 = torch.norm(original_features - transformed_features, p=2, dim=1)
                cosine = torch.nn.functional.cosine_similarity(
                    original_features, transformed_features, dim=1
                )

                total_l1.append(l1)
                total_l2.append(l2)
                total_cosine.append(cosine)

                pbar.update(1)

    # Compute summary statistics
    mean_l1 = torch.cat(total_l1).mean().item()
    mean_l2 = torch.cat(total_l2).mean().item()
    mean_cosine = torch.cat(total_cosine).mean().item()

    print(f"L1 distance: {mean_l1:.4f}")
    print(f"L2 distance: {mean_l2:.4f}")
    print(f"Cosine similarity: {mean_cosine:.4f}")

    results = {
        "l1_distance": mean_l1,
        "l2_distance": mean_l2,
        "cosine_similarity": mean_cosine,
    }

    # Log summary statistics
    if logging:
        wandb.log(results)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute feature distances")
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
        "--passes",
        "-p",
        type=int,
        required=False,
        default=1,
        help="Number of passes of the dataset.",
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
        "--label",
        "-l",
        type=str,
        required=False,
        default="dummy",
        help="Dataset label to use.",
    )
    parser.add_argument(
        "--nolog",
        action="store_true",
        help="Do not log to wandb.",
    )

    args = parser.parse_args()

    feature_distances(
        feature=args.feature,
        dataset=args.dataset,
        transform=args.transform,
        passes=args.passes,
        batch_size=args.batch_size,
        device=args.device,
        logging=not args.nolog,
        label=args.label,
    )
