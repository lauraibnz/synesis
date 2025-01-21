import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
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
        sr = feature_config["sample_rate"]
        transform_obj = get_transform(transform_config, sr)

    loader = DataLoader(raw_dataset, shuffle=False, batch_size=batch_size)

    results_file = Path(f"results/{dataset}_{transform}_feature_distances.csv")
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
                    for j in range(batch_size):
                        transformed_raw_data.append(transform_obj(items[j]))
                        if transform == "AddReverb":
                            transform_params.append(
                                transform_obj.parameters["target_rt60"]
                            )
                        else:
                            raise ValueError(
                                f"Don't know what parameter to use for: {transform}"
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
        "--device",
        type=str,
        required=False,
        help="Device to use for training.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        required=False,
        default=32,
        help="Batch size.",
    )

    args = parser.parse_args()

    results = feature_distances(
        feature=args.feature,
        dataset=args.dataset,
        transform=args.transform,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Load CSV
    df = pd.read_csv(f"results/{args.dataset}_{args.feature}_feature_distances.csv")

    # Create scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="transform_param", y="l2_distance")

    # Add labels and title
    plt.xlabel("Transform Parameter")
    plt.ylabel("L2 Distance")
    plt.title("Scatterplot of Transform Parameter vs L2 Distance")

    # Show plot
    plt.show()
    plt.savefig(f"./scatterplot_{args.dataset}_{args.feature}_{args.transform}.png")
