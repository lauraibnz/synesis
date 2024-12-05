import pytest
import torch
from torch import nn

from synesis.downstream import evaluate, train


@pytest.fixture(params=["vggish_mtat"])
def mock_feature_name(request):
    return request.param


@pytest.fixture(params=["TinySOL"])
def dataset_name(request):
    return request.param


@pytest.fixture(params=["pitch_classification"])
def task_name(request):
    return request.param


@pytest.fixture(params=[True])
def feature_aggregation(request):
    return request.param


@pytest.fixture(params=["feature"])
def item_format(request):
    return request.param


def test_train_model(
    dataset_name,
    task_name,
    item_format,
    mock_feature_name,
    feature_aggregation,
    tmp_path,
):
    # Override config with minimal settings
    task_config = {
        "training": {
            "batch_size": 16,
            "num_epochs": 2,
            "feature_aggregation": feature_aggregation,
        }
    }

    # Train model
    model = train(
        feature=mock_feature_name,
        dataset=dataset_name,
        task=task_name,
        task_config=task_config,
        item_format=item_format,
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


def test_evaluate_model(
    dataset_class, task_name, item_format, mock_feature_name, device
):
    # Configure minimal evaluation settings
    test_config = {"evaluation": {"batch_size": 2}}

    # First train a model with minimal epochs
    model = train(
        feature=mock_feature_name,
        dataset=dataset_class.__name__,
        task=task_name,
        task_config={"training": {"num_epochs": 1}},
        device=device,
        item_format=item_format,
    )

    # Evaluate the model
    results = evaluate(
        model=model,
        feature=mock_feature_name,
        dataset=dataset_class.__name__,
        task=task_name,
        task_config=test_config,
        device=device,
        item_format=item_format,
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


def test_model_save_load(
    dataset_class, task_name, item_format, mock_feature_name, tmp_path, device
):
    # Train model with minimal epochs
    model = train(
        feature=mock_feature_name,
        dataset=dataset_class.__name__,
        task=task_name,
        task_config={"training": {"num_epochs": 1}},
        device=device,
        item_format=item_format,
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


if __name__ == "__main__":
    pytest.main()
