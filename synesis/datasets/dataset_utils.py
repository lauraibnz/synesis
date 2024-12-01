import importlib
from typing import Type

from torch import tensor
from torch.utils.data import Dataset


class DatasetFactory:
    @classmethod
    def get_dataset(cls, name: str, **kwargs) -> Dataset:
        """
        Retrieve a dataset instance based on its name and optional parameters.

        Args:
            name: The name of the dataset to retrieve.
            **kwargs: Optional parameters to pass to the dataset constructor.

        Returns:
            An instance of the requested dataset.

        Raises:
            ValueError: If the dataset name is not recognized.
        """
        try:
            # Dynamically import the dataset module
            module = importlib.import_module(f"synesis.datasets.{name.lower()}")
            # Get the dataset class from the module
            dataset_class: Type[Dataset] = getattr(module, name)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Unknown dataset: {name}") from e

        # Get the default parameter values from the dataset class
        default_params = {
            k: v.default
            for k, v in dataset_class.__init__.__annotations__.items()
            if k != "return" and hasattr(v, "default")
        }

        # Update default parameters with provided kwargs
        params = {**default_params, **kwargs}

        return dataset_class(**params)


class SubitemDataset(Dataset):
    """
    Wrapper for datasets that normally return items with variable
    number of subitems, such as with audio datasets. Provides
    __getitem__ and __len__ methods that use subitem indices.

    Args:
        dataset: The dataset to wrap.

    Returns:
        A dataset that returns subitems instead of items.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        self.item_lengths = [len(item[0]) for item in dataset]
        self.real_len = sum(self.item_lengths)
        self.real_indices = list(range(self.real_len))

        # subarray indexing to (array, relative offset) tuple
        self.idx_map = {}
        real_idx = 0
        for i, item_len in enumerate(self.item_lengths):
            for j in range(item_len):
                self.idx_map[real_idx] = (i, j)
                real_idx += 1

    def __len__(self):
        return self.real_len

    def __getitem__(self, idx):
        array_idx, offset = self.idx_map[idx]
        array, label = self.dataset[array_idx]
        array = tensor(array[offset])

        return array.squeeze(0), label


def get_dataset(name: str, **kwargs) -> Dataset:
    """
    A convenience function to get a dataset instance.

    This function is a wrapper around DatasetFactory.get_dataset for
    easier importing and usage.

    Args:
        name: The name of the dataset to retrieve.
        **kwargs: Optional parameters to pass to the dataset constructor.

    Returns:
        An instance of the requested dataset.
    """
    return DatasetFactory.get_dataset(name, **kwargs)
