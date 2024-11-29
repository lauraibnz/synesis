from pathlib import Path

import pytest
import torch
from torch import nn

from config.tasks import task_configs
from synesis.datasets.tinysol import TinySOL
from synesis.downstream import evaluate, train


@pytest.fixture(params=[TinySOL])
def dataset_class(request):
    return request.param


@pytest.fixture(params=["instrument_classification"])
def task_name(request):
    return request.param


@pytest.fixture
def mock_feature_name():
    return "vggish_mtat"  # Use a simple feature extractor for testing


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_train_model(dataset_class, task_name, mock_feature_name, tmp_path, device):
    # Configure a minimal training run for testing
    test_config = {"training": {"num_epochs": 2, "batch_size": 2, "patience": 3}}

    # Create small test dataset
    dataset = dataset_class(
        feature=mock_feature_name,
        root=f"data/{dataset_class.__name__}",
        split="train",
    )

    # Take small subset for testing
    subset_size = min(10, len(dataset))
    dataset.paths = dataset.paths[:subset_size]
    dataset.labels = dataset.labels[:subset_size]

    # Train model
    model = train(
        feature=mock_feature_name,
        dataset=dataset_class.__name__,
        task=task_name,
        task_config=test_config,
        device=device,
    )

    # Basic assertions
    assert isinstance(model, nn.Module), "Model should be a PyTorch module"
    assert not model.training, "Model should be in eval mode after training"

    # Check if model weights exist and are not all zero
    has_nonzero_params = False
    for param in model.parameters():
        if torch.any(param != 0):
            has_nonzero_params = True
            break
    assert has_nonzero_params, "Model parameters should not all be zero"


def test_evaluate_model(dataset_class, task_name, mock_feature_name, device):
    # Configure minimal evaluation settings
    test_config = {"evaluation": {"batch_size": 2}}

    # First train a model with minimal epochs
    model = train(
        feature=mock_feature_name,
        dataset=dataset_class.__name__,
        task=task_name,
        task_config={"training": {"num_epochs": 1}},
        device=device,
    )

    # Evaluate the model
    results = evaluate(
        model=model,
        feature=mock_feature_name,
        dataset=dataset_class.__name__,
        task=task_name,
        task_config=test_config,
        device=device,
    )

    # Check evaluation results
    assert isinstance(results, dict), "Evaluation should return a dictionary of metrics"
    assert len(results) > 0, "Should have at least one metric result"

    # Check that metric values are reasonable
    for metric_name, value in results.items():
        assert isinstance(
            value, (float, int)
        ), f"Metric {metric_name} should be numeric"
        assert not torch.isnan(
            torch.tensor(value)
        ), f"Metric {metric_name} should not be NaN"


def test_model_save_load(dataset_class, task_name, mock_feature_name, tmp_path, device):
    # Train model with minimal epochs
    model = train(
        feature=mock_feature_name,
        dataset=dataset_class.__name__,
        task=task_name,
        task_config={"training": {"num_epochs": 1}},
        device=device,
    )

    # Save model
    save_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), save_path)

    # Check if file exists
    assert save_path.exists(), "Model file should exist after saving"

    # Load model
    loaded_state_dict = torch.load(save_path)
    new_model = type(model)(*model.init_args).to(device)
    new_model.load_state_dict(loaded_state_dict)

    # Compare original and loaded models
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2), "Loaded model parameters should match original"


@pytest.mark.parametrize("item_format", ["audio", "feature"])
def test_different_input_formats(
    item_format, dataset_class, task_name, mock_feature_name, device
):
    # Test both audio and feature input formats
    test_config = {"training": {"num_epochs": 1, "batch_size": 2}}

    model = train(
        feature=mock_feature_name,
        dataset=dataset_class.__name__,
        task=task_name,
        item_format=item_format,
        task_config=test_config,
        device=device,
    )

    assert isinstance(
        model, nn.Module
    ), f"Model training failed for {item_format} format"


if __name__ == "__main__":
    pytest.main()
