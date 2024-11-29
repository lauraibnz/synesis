from typing import Dict, Type

from torch.utils.data import Dataset

from synesis.datasets.magnatagatune import MagnaTagATune
from synesis.datasets.mtgjamendo import MTGJamendo
from synesis.datasets.tinysol import TinySOL


class DatasetFactory:
    _datasets: Dict[str, Type[Dataset]] = {
        "magnatagatune": MagnaTagATune,
        "mtgjamendo": MTGJamendo,
        "tinysol": TinySOL,
    }

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
        dataset_class = cls._datasets.get(name.lower())
        if dataset_class is None:
            raise ValueError(f"Unknown dataset: {name}")

        # Get the default parameter values from the dataset class
        default_params = {
            k: v.default
            for k, v in dataset_class.__init__.__annotations__.items()
            if k != "return" and hasattr(v, "default")
        }

        # Update default parameters with provided kwargs
        params = {**default_params, **kwargs}

        return dataset_class(**params)


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
