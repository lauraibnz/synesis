import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.features import configs as feature_configs
from config.transforms import configs as transform_configs
from synesis.datasets.dataset_utils import SubitemDataset, get_dataset
from synesis.features.feature_utils import get_feature_extractor
from synesis.transforms.transform_utils import get_transform


def feature_distances(
    feature: str,
    dataset: str,
    transform: str,
    batch_size: int = 32,
    passes: int = 3,
    device: Optional[str] = None,
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
    if dataset not in ["ImageNet", "LibriSpeech"]:
        raise ValueError(f"Invalid dataset: {dataset}")

    feature_config = feature_configs.get(feature)
    transform_config = transform_configs.get(transform)

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the dataset
    raw_dataset = get_dataset(
        name=dataset,
        feature=feature,
        transform=transform,
        label="dummy",
        split="test",
        download=False,
        item_format="raw",
    )

    # If dataset returns subitems per item, need to wrap it
    if dataset != "ImageNet" and raw_dataset[0][0].dim() == 3:
        wrapped_dataset = SubitemDataset(raw_dataset)
        del raw_dataset
        raw_dataset = wrapped_dataset

    feature_extractor = get_feature_extractor(feature)
    feature_extractor = feature_extractor.to(device)

    if dataset == "ImageNet":
        transform_obj = None
    elif dataset == "LibriSpeech":
        transform_obj = get_transform(transform_config)

    loader = DataLoader(raw_dataset, shuffle=False, batch_size=batch_size)

    results_file = Path(
        f"results/{dataset}_{feature}_{transform}_feature_distances.csv"
    )
    if not results_file.exists():
        pd.DataFrame(
            columns=[
                "transform_param",
                "l1_distance",
                "l2_distance",
                "cosine_similarity",
            ]
        ).to_csv(results_file, index=False)

    for pass_ in range(passes):
        with tqdm(total=len(loader), desc=f"Pass {pass_+1}/{passes}") as pbar:
            for i, (items, labels) in enumerate(loader):
                # get original and transformed data, along with transform parameters
                if dataset == "ImageNet":
                    original_raw_data = items[:, 0].to(device)
                    transformed_raw_data = items[:, 1].to(device)
                    transform_params = labels.to(device)
                elif dataset == "LibriSpeech":
                    original_raw_data = items.to(device)
                    transform_params = []
                    transformed_raw_data = []
                    for j in range(original_raw_data.shape[0]):
                        if transform == "AddReverb":
                            audio_numpy = items[j].cpu().numpy().astype(np.float32)
                            max_retries = 5
                            for retry in range(max_retries):
                                try:
                                    # room might be too large for target rt60, so
                                    # keep trying until you get a large enough value
                                    transformed_audio = transform_obj(
                                        audio_numpy,
                                        sample_rate=feature_config["sample_rate"],
                                    )
                                    transformed_raw_data.append(
                                        torch.from_numpy(transformed_audio)
                                    )
                                    transform_params.append(
                                        transform_obj.parameters["target_rt60"]
                                    )
                                    break  # Success - exit retry loop
                                except ValueError as e:
                                    if retry == max_retries - 1:  # Last attempt
                                        raise RuntimeError(
                                            f"Failed after {max_retries} attempts: {e}"
                                        )
                                    continue  # Try again with new random parameters
                        else:
                            raise ValueError(
                                f"Transform not fully implemented: {transform}"
                            )
                    transformed_raw_data = torch.stack(transformed_raw_data).to(device)
                    transform_params = torch.tensor(transform_params).to(device)

                # extract features
                original_features = feature_extractor(original_raw_data)
                transformed_features = feature_extractor(transformed_raw_data)

                # calculate L1, L2, and cosine distances
                l1 = torch.norm(original_features - transformed_features, p=1)
                l2 = torch.norm(original_features - transformed_features, p=2)
                cosine = torch.nn.functional.cosine_similarity(
                    original_features, transformed_features
                )

                # Save batch results
                batch_results = pd.DataFrame(
                    {
                        "transform_param": transform_params.cpu().numpy(),
                        "l1_distance": l1.cpu().numpy(),
                        "l2_distance": l2.cpu().numpy(),
                        "cosine_similarity": cosine.cpu().numpy(),
                    }
                )
                batch_results.to_csv(results_file, mode="a", header=False, index=False)

                pbar.update(1)

    results = pd.read_csv(results_file)
    return results


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

    args = parser.parse_args()

    results = feature_distances(
        feature=args.feature,
        dataset=args.dataset,
        transform=args.transform,
        passes=args.passes,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Load CSV
    df = pd.read_csv(
        f"results/{args.dataset}_{args.feature}_{args.transform}_feature_distances.csv"
    )

    # Create scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="transform_param", y="cosine_similarity")

    # Add labels and title
    plt.xlabel("Transform Parameter")
    plt.ylabel("Cosine similarity")
    plt.title("Scatterplot of Transform Parameter vs Cosine Similarity")

    # Show plot
    plt.show()
    plt.savefig(
        f"results/{args.dataset}_{args.feature}_{args.transform}_cosine_similarity.png"
    )
