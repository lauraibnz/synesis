from pathlib import Path

import numpy as np
import pytest
import torch

from config.features import feature_config
from ref.datasets.magnatagatune import MagnaTagATune
from ref.datasets.mtgjamendo import MTGJamendo
from ref.features.feature_utils import dynamic_batch_extractor, get_pretrained_model


@pytest.fixture(params=[MagnaTagATune, MTGJamendo])
def dataset_sample(request):
    DatasetClass = request.param
    dataset = DatasetClass(
        feature="vggish_mtat",
        root=f"data/{DatasetClass.__name__}",
        item_format="audio",
        split=None,
    )
    # Take a small subset of paths for testing
    subset_size = min(5, len(dataset))
    dataset.paths = dataset.paths[:subset_size]
    dataset.labels = dataset.labels[:subset_size]
    return dataset


def test_feature_extraction(dataset_sample, tmp_path):
    # Set up temporary output directory
    output_dir = tmp_path / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update feature_paths to use the temporary directory
    dataset_sample.feature_paths = [
        output_dir / f"{Path(path).stem}.npy" for path in dataset_sample.paths
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the pretrained model
    model = get_pretrained_model("vggish_mtat")

    # Test if model is in eval mode
    assert not model.training, "Model should be in eval mode"

    # Get feature config
    config = feature_config["vggish_mtat"]
    item_len_samples = int(config["item_len_sec"] * config["sample_rate"])

    # Perform feature extraction
    dynamic_batch_extractor(
        dataset_sample,
        model,
        item_len=item_len_samples,
        padding="repeat",
        batch_size=2,
        device=device,
    )

    # Check if features were extracted and saved correctly
    for feature_path in dataset_sample.feature_paths:
        assert feature_path.exists(), f"Feature file {feature_path} not created"

        # Load the extracted feature
        with feature_path.open("rb") as f:
            feature = np.load(f)

        # Check feature shape
        assert feature.ndim == 2, f"Feature {feature_path} should be 2-dimensional"
        assert (
            feature.shape[1] == 512
        ), f"Feature {feature_path} should have 512 dimensions"


if __name__ == "__main__":
    pytest.main()
