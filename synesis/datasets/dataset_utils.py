import importlib
from pathlib import Path
from typing import Type, Union

import torch
import torchaudio
from torch import Tensor, tensor
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

        # for accessing dataset info easier/consistently
        self.label_encoder = dataset.label_encoder

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

        return array, label


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


def load_track(
    path: Union[str, Path],
    item_format: str,
    itemization: bool,
    item_len_sec: float,
    sample_rate: int,
) -> Tensor:
    """Load an audio track (or features for it) from a file.

    Args:
        path: The path to the audio file.
        item_format: Format of the items to return: ["raw", "feature"].
        itemization: For datasets with variable-length items, whether to return them
                  as a list of equal-length items (True) or as a single item.
        item_len_sec: The length of the items in seconds.
        sample_rate: The sample rate to resample the audio to.
    """
    if item_format == "feature":
        feature = torch.load(path, weights_only=False)
        # assumes there wasn't already a channel dim, but it's hard
        # to check otherwise...
        feature = feature.unsqueeze(1)
        if not itemization:
            # concatenate
            feature = feature.view(-1, 1, feature.size(2))
        return feature
    else:
        waveform, original_sample_rate = torchaudio.load(path, normalize=True)
        if waveform.size(0) != 1:  # make mono if stereo (or more)
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate,
                new_freq=sample_rate,
            )
            waveform = resampler(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if itemization:
            # we need to split the track into fixed-length segments of
            # item_len_sec and return them as a list
            item_len_samples = int(item_len_sec * sample_rate)
            waveform_len = waveform.size(1)
            num_items = waveform_len // item_len_samples
            remainder_item = waveform_len % item_len_samples

            # pad with zeros so that we can split without remainder
            waveform = torch.cat(
                [waveform, torch.zeros(1, item_len_samples - remainder_item)],
                dim=1,
            )

            # split into subitems of size item_len_samples
            waveform = waveform.view(num_items + 1, 1, item_len_samples)

        return waveform
