import argparse
from typing import Optional

import torch

from config.features import feature_configs
from synesis.datasets.dataset_utils import get_dataset
from synesis.features.feature_utils import dynamic_batch_extractor, get_pretrained_model


def extract_features(
    feature: str,
    dataset: str,
    batch_size: int = 32,
    device: Optional[str] = None,
    download_dataset: bool = False,
):
    """Extract features from audio files.

    Args:
        feature: Feature extraction model name.
        dataset: Dataset name.
        batch_size: Batch size for feature extraction.
        feature_config: Configuration for the feature extractor. If None,
                        default configuration is used.
        device: Device to use for feature extraction.
                If None, GPU is used if available.
    """

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = get_pretrained_model(feature)

    dataset = get_dataset(
        name=dataset,
        feature=feature,
        item_format="audio",
        split=None,  # Use full dataset for feature extraction
        download=download_dataset,
        itemization=False,  # dynamic extractor will handle itemization
    )

    # attempt to get item length directly in samples, if
    # it doesn't exist, calculate it from item_len_sec and sr
    item_len = feature_configs[feature].get(
        "item_len",
        int(
            feature_configs[feature]["item_len_sec"]
            * feature_configs[feature]["sample_rate"]
        ),
    )

    dynamic_batch_extractor(
        dataset=dataset,
        extractor=extractor,
        item_len=item_len,
        batch_size=batch_size,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio files.")
    parser.add_argument(
        "--feature",
        "-f",
        type=str,
        required=True,
        help="Feature extraction model name.",
    )
    parser.add_argument(
        "--dataset", "-d", type=str, required=True, help="Dataset name."
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use for feature extraction."
    )
    args = parser.parse_args()

    extract_features(
        feature=args.feature,
        dataset=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )
