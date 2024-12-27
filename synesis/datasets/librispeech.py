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
            datasets = []
            for split_name in split_map.values():
                datasets.append(
                    torchaudio.datasets.LIBRISPEECH(
                        root=str(self.root).replace("LibriSpeech", ""),
                        url=split_name,
                        download=download,
                    )
                )
            self.dataset = ConcatDataset(datasets)
        else:
            self.dataset = torchaudio.datasets.LIBRISPEECH(
                root=str(self.root).replace("LibriSpeech", ""),
                url=split_map[split] if split else None,
                download=download,
            )

        self._load_metadata()

    def _load_metadata(self) -> None:
        # load audio paths
        self.paths = [
            str(
                Path(self.dataset._path)
                / self.dataset._walker[i].split("-")[0]
                / self.dataset._walker[i].split("-")[1]
                / f"{self.dataset._walker[i]}.{self.audio_format}"
            )
            for i in range(len(self.dataset))
        ]

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
            # get transcript path
            transcript_file_name = (
                "-".join(self.paths[idx].split("/")[-1].split("-")[:-1]) + ".trans.txt"
            )
            transcript_path = Path(self.paths[idx]).parent / transcript_file_name
            transcript_id = self.paths[idx].split("/")[-1].split(".")[0]

            # calculate words from the transcript
            transcript = ""
            with open(transcript_path, "r") as f:
                # get line that starts with transcript_id
                for line in f:
                    if line.startswith(transcript_id):
                        transcript = line.split(" ", 1)[1].strip()
                        break
            if not transcript:
                warnings.warn(f"Transcript not found for {transcript_id}")

            words = len(transcript.split())
            audio_len = get_duration(path=self.paths[idx])
            wps = words / audio_len
            target = torch.tensor(wps, dtype=torch.float32)
        else:
            target = torch.tensor(0, dtype=torch.float32)

        return y, target
