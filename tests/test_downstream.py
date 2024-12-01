import pytest
import torch
from torch import nn, tensor
from torch.utils.data import Dataset

from config.tasks import task_configs
from synesis.datasets.dataset_utils import SubitemDataset
from synesis.datasets.tinysol import TinySOL
from synesis.downstream import evaluate, train


@pytest.fixture(params=[TinySOL])
def dataset_class(request):
    return request.param


@pytest.fixture(params=["instrument_classification"])
def task_name(request):
    return request.param


@pytest.fixture(params=["feature"])
def item_format(request):
    return request.param


@pytest.fixture
def mock_feature_name():
    return "vggish_mtat"  # Use a simple feature extractor for testing


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_subitem_wrapper():
    # Create a simple dataset with a few items
    class MockDataset(Dataset):
        def __init__(self):
            self.items = [
                [
                    tensor([0, 0, 0]).unsqueeze(0),
                    tensor([1, 1, 1]).unsqueeze(0),
                    tensor([2, 2, 2]).unsqueeze(0),
                ],
                [
                    tensor([3, 3, 3]).unsqueeze(0),
                    tensor([4, 4, 4]).unsqueeze(0),
                ],
                [
                    tensor([5, 5, 5]).unsqueeze(0),
                ],
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx], "mock_label"

    dataset = MockDataset()
    subitem_dataset = SubitemDataset(dataset)

    dataloader = torch.utils.data.DataLoader(subitem_dataset, batch_size=4)

    for i, (item, label) in enumerate(dataloader):
        assert len(item) <= 4, "Batch size should not exceed 4"
        assert len(item[0]) == 3, "Feature dimension should be 3"
        assert label[0] == "mock_label", "Label should be the same for all items"
        if i == 0:
            assert torch.equal(
                item,
                torch.tensor(
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [2, 2, 2],
                        [3, 3, 3],
                    ]
                ),
            )
        elif i == 1:
            assert torch.equal(
                item,
                torch.tensor(
                    [
                        [4, 4, 4],
                        [5, 5, 5],
                    ]
                ),
            )


def test_train_model(
    dataset_class, task_name, item_format, mock_feature_name, tmp_path, device
):
    # Configure a minimal training run for testing
    test_config = {"training": {"num_epochs": 2, "batch_size": 2, "patience": 3}}

    # Create small test dataset
    dataset = dataset_class(
        feature=mock_feature_name,
        root=f"data/{dataset_class.__name__}",
        split="train",
        item_format=item_format,
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
