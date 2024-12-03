import numpy as np
import pytest
import torch
from torch import tensor
from torch.utils.data import Dataset

from synesis.dataset.dataset_utils import SubitemDataset
from synesis.datasets.magnatagatune import MagnaTagATune
from synesis.datasets.mtgjamendo import MTGJamendo
from synesis.datasets.tinysol import TinySOL

DATASETS = [
    (
        MagnaTagATune,
        {
            "root": "data/MagnaTagATune",
            "splits": [None, "train", "test", "validation"],
            "item_format": "audio",
        },
    ),
    (
        MTGJamendo,
        {
            "root": "data/MTGJamendo",
            "splits": [None, "train", "test", "validation"],
            "subsets": [None, "top50tags", "genre", "instrument", "moodtheme"],
            "item_format": "audio",
        },
    ),
    (
        TinySOL,
        {
            "root": "data/TinySOL",
            "splits": [None, "train", "test", "validation"],
            "item_format": "audio",
        },
    ),
]


@pytest.fixture(params=DATASETS)
def dataset_config(request):
    return request.param


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


def test_dataset_loading(dataset_config):
    DatasetClass, config = dataset_config
    for split in config["splits"]:
        dataset = DatasetClass(
            feature="vggish_mtat",
            root=config["root"],
            item_format=config["item_format"],
            split=split,
        )

        # Test paths and labels are loaded
        assert len(dataset.paths) > 0
        assert len(dataset.audio_paths) > 0
        assert len(dataset.feature_paths) > 0
        assert len(dataset.labels) > 0
        assert len(dataset.paths) == len(dataset.labels)
        assert dataset.labels.dtype == torch.long

        # Test random items
        for _ in range(5):
            idx = np.random.randint(0, len(dataset))
            item, label = dataset[idx]
            assert len(item.shape) == 2
            assert torch.is_tensor(item)
            if dataset.__class__.__name__ in ["MagnaTagATune", "MTGJamendo"]:
                assert isinstance(label, torch.Tensor)
                assert label.dtype == torch.long


def test_mtgjamendo_subsets():
    config = next(conf for cls, conf in DATASETS if cls == MTGJamendo)
    for subset in config["subsets"]:
        for split in config["splits"]:
            dataset = MTGJamendo(
                feature="vggish_mtat",
                root=config["root"],
                item_format=config["item_format"][0],
                split=split,
                subset=subset,
            )
            assert (
                len(dataset) > 0
            ), f"Empty dataset for subset: {subset}, split: {split}"
            if hasattr(dataset, "metadata_path"):
                if subset:
                    assert str(subset) in str(
                        dataset.metadata_path
                    ), f"Subset {subset} not in metadata path for split {split}"
                if split:
                    assert str(split) in str(
                        dataset.metadata_path
                    ), f"Split {split} not in metadata path for subset {subset}"


if __name__ == "__main__":
    pytest.main()
