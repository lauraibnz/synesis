from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from config.features import configs as feature_configs
from synesis.datasets.dataset_utils import load_track


class LibriSpeech(Dataset):
    def __init__(
        self,
        feature: str,
        root: Union[str, Path] = "data/LibriSpeech",
        split: Optional[str] = None,
        download: bool = False,
        feature_config: Optional[dict] = None,
        audio_format: str = "flac",
        item_format: str = "feature",
        itemization: bool = True,
        seed: int = 42,
    ) -> None:
        """
        LibriSpeech dataset implementation.

        Args:
            feature: If split is None, prepare dataset for this feature extractor.
                        If split is not None, load these extracted features.
            root: Root directory of the dataset. Defaults to "data/LibriSpeech".
            split: Split of the dataset to use: None for all, 'train' for 'train-100',
                   'validation' for 'dev', 'test' for 'test-clean'.
            download: Whether to download the dataset if it doesn't exist.
            feature_config: Configuration for the feature extractor.
            audio_format: Format of the audio files: ["flac", "wav"].
            item_format: Format of the items to return: ["raw", "feature"].
            seed: Random seed for reproducibility.
        """
        root = Path(root)
        self.root = root
        if split not in [None, "train", "validation", "test"]:
            raise ValueError(
                f"Invalid split: {split} "
                + "Options: None, 'train', 'validation', 'test'"
            )
        self.split = split
        self.item_format = item_format
        self.itemization = itemization
        self.audio_format = audio_format
        self.feature = feature

        if not feature_config:
            # load default feature config
            feature_config = feature_configs[feature]
        self.feature_config = feature_config

        # initialize torchaudio dataset
        split_map = {
            "train": "train-clean-100",
            "validation": "dev-clean",
            "test": "test-clean",
        }
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(self.root),
            url=split_map[split] if split else None,
            download=download,
        )

        self._load_metadata()

    def _load_metadata(self) -> None:
        # load audio paths
        self.paths = [self.dataset._walker[i] for i in range(len(self.dataset))]

        self.feature_paths = [
            path.replace(f".{self.audio_format}", ".pt")
            .replace(f"/{self.audio_format}/", f"/{self.feature}/")
            .replace("/audio/", f"/{self.feature}/")
            for path in self.paths
        ]
        self.raw_data_paths = self.paths
        self.paths = (
            self.raw_data_paths if self.item_format == "raw" else self.feature_paths
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        path = (
            self.raw_data_paths[idx]
            if self.item_format == "raw"
            else self.feature_paths[idx]
        )

        track = load_track(
            path=path,
            item_format=self.item_format,
            itemization=self.itemization,
            item_len_sec=self.feature_config["item_len_sec"],
            sample_rate=self.feature_config["sample_rate"],
        )

        return track, torch.tensor([])  # Return an empty tensor for labels


if __name__ == "__main__":
    librispeech = LibriSpeech(
        feature="VGGishMTAT",
        root="data/LibriSpeech",
        split=None,
        item_format="raw",
    )
    # iterate over all items
    import numpy as np

    for _ in range(5):
        idx = np.random.randint(0, len(librispeech))
        item, label = librispeech[idx]
        print(item.shape, label)
