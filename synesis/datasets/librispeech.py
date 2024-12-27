import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchaudio
from librosa import get_duration
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

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
        fv: str = "wps",
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
            itemization: Whether to return the full item or itemized parts.
            fv: factor of variations (i.e. label) to return
                e.g. "wps" for words per second
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
        self.fv = fv

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
        if split is None:
            # we need to load all splits and concatenate them
            self.datasets = []
            for split_name in split_map.values():
                self.datasets.append(
                    torchaudio.datasets.LIBRISPEECH(
                        root=str(self.root).replace("LibriSpeech", ""),
                        url=split_name,
                        download=download,
                    )
                )
            self.dataset = ConcatDataset(self.datasets)
        else:
            self.datasets = None
            self.dataset = torchaudio.datasets.LIBRISPEECH(
                root=str(self.root).replace("LibriSpeech", ""),
                url=split_map[split] if split else None,
                download=download,
            )

        self._load_metadata()

    def _get_dataset_and_index(self, idx: int):
        """Maps global index to (dataset, local_index) for ConcatDataset."""
        if not self.datasets:
            return self.dataset, idx

        dataset_idx = 0
        while idx >= len(self.datasets[dataset_idx]):
            idx -= len(self.datasets[dataset_idx])
            dataset_idx += 1
        return self.datasets[dataset_idx], idx

    def _load_metadata(self) -> None:
        self.paths = []
        if self.datasets:
            # Handle concatenated datasets
            for dataset in self.datasets:
                dataset_path = Path(dataset._path)
                for i in range(len(dataset)):
                    file_id = dataset._walker[i]
                    speaker_id, chapter_id = file_id.split("-")[:2]
                    self.paths.append(
                        str(
                            dataset_path
                            / speaker_id
                            / chapter_id
                            / f"{file_id}.{self.audio_format}"
                        )
                    )
        else:
            # Handle single dataset
            dataset_path = Path(self.dataset._path)
            for i in range(len(self.dataset)):
                file_id = self.dataset._walker[i]
                speaker_id, chapter_id = file_id.split("-")[:2]
                self.paths.append(
                    str(
                        dataset_path
                        / speaker_id
                        / chapter_id
                        / f"{file_id}.{self.audio_format}"
                    )
                )

        self.feature_paths = [
            path.replace(f".{self.audio_format}", ".pt").replace(
                "/LibriSpeech/", f"/LibriSpeech/{self.feature}/"
            )
            for path in self.paths
        ]
        self.raw_data_paths = self.paths
        self.paths = (
            self.raw_data_paths if self.item_format == "raw" else self.feature_paths
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        y = load_track(
            path=self.paths[idx],
            item_format=self.item_format,
            itemization=self.itemization,
            item_len_sec=self.feature_config["item_len_sec"],
            sample_rate=self.feature_config["sample_rate"],
        )

        if self.fv == "wps":
            # Use pathlib for cleaner path handling
            path = Path(self.paths[idx])
            file_id = path.stem  # Get filename without extension
            transcript_file_name = "-".join(file_id.split("-")[:-1]) + ".trans.txt"
            transcript_path = path.parent / transcript_file_name
            transcript_id = file_id

            # calculate words from the transcript
            transcript = ""
            with open(transcript_path, "r") as f:
                for line in f:
                    if line.startswith(transcript_id):
                        transcript = line.split(" ", 1)[1].strip()
                        break
            if not transcript:
                warnings.warn(f"Transcript not found for {transcript_id}")

            words = len(transcript.split())
            audio_len = get_duration(path=str(path))
            wps = words / audio_len
            target = torch.tensor(wps, dtype=torch.float32)
        else:
            target = torch.tensor(0, dtype=torch.float32)

        return y, target
