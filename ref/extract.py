import torch
import argparse
from typing import Optional
from ref.features.feature_utils import get_pretrained_model, smart_batch_processor
from ref.datasets.dataset_utils import get_dataset
from config.features import feature_config as fc


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

    feature_config = fc[feature]

    extractor = get_pretrained_model(feature)
    extractor.to(device)

    dataset = get_dataset(
        name=dataset,
        feature=feature,
        split=None,  # Use full dataset for feature extraction
        download=download_dataset,
    )

    smart_batch_processor(
        dataset=dataset,
        extractor=extractor,
        item_len=feature_config["item_len"],
        batch_size=batch_size,
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
