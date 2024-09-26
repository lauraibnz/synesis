import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
from torch.utils.data import Dataset

from config.features import feature_config as fc
from ref.datasets.dataset_utils import download_github_dir, download_github_file


class MTGJamendo(Dataset):
    def __init__(
        self,
        feature: str,
        root: Union[str, Path] = "data/MTGJamendo",
        subset: Optional[str] = None,
        split: Optional[str] = None,
        download: bool = False,
        feature_config: Optional[dict] = None,
        seed: int = 42,
    ) -> None:
        """MTG Jamendo (and subset) dataset implementation.

        Args:
            feature: If split is None, prepare dataset for this feature extractor.
                     If split is not None, load these extracted features.
            root: Root directory of the dataset. Defaults to "data/MTGJamendo".
            subset: Subset of the dataset to use: ["top50tags", "genre", "instrument",
                    "moodtheme", None], where None uses the full dataset.
            split: Split of the dataset to use: ["train", "test", "val", None], where
                     None uses the full dataset (e.g. for feature extraction).
            download: Whether to download the dataset if it doesn't exist.
            feature_config: Configuration for the feature extractor.
            seed: Random seed for reproducibility.
        """
        root = Path(root)
        self.root = root
        if subset not in [None, "top50tags", "genre", "instrument", "moodtheme"]:
            raise ValueError(
                f"Invalid subset: {subset} "
                + "Options: None, 'top50tags', 'genre', 'instrument', 'moodtheme'"
            )
        if split not in [None, "train", "test", "validation"]:
            raise ValueError(
                f"Invalid split: {split} " + "Options: None, 'train', 'test', 'validation'"
            )
        self.subset = subset
        self.split = split

        if download:
            self._download()

        if not self.split:
            # pretrain mode, so load all paths for each subset
            if self.subset:
                self.metadata_path = root / "metadata" / f"autotagging_{subset}.tsv"
            else:
                # load full, cleaned-up dataset
                self.metadata_path = (
                    root / "metadata" / "raw_30s_cleantags_50artists.tsv"
                )
        else:
            # train, test, or validation splits for downstream
            self.metadata_path = root / "metadata" / f"autotagging_{subset}-{split}.tsv"

        if not feature_config:
            # load default feature config
            feature_config = fc[feature]

        self.mlb = MultiLabelBinarizer()

        self.paths, self.labels = self._load_metadata()

    def _download(self) -> None:
        warnings.warn(
            "Audio download not implemented yet, downloading metadata only.",
            stacklevel=2,
        )
        download_github_dir(
            owner="MTG",
            repo="mtg-jamendo-dataset",
            path="data/splits/split-0",
            save_dir=str(self.root / "metadata"),
        )
        for f in [
            "autotagging.tsv",
            "autotagging_genre.tsv",
            "autotagging_instrument.tsv",
            "autotagging_moodtheme.tsv",
            "raw_30s_cleantags_50artists.tsv",
        ]:
            download_github_file(
                owner="MTG",
                repo="mtg-jamendo-dataset",
                file_path=f"data/{f}",
                save_dir=str(self.root / "metadata"),
            )

    def _load_metadata(self) -> Tuple[list, np.ndarray]:
        with open(self.metadata_path, "r") as f:
            metadata = f.readlines()
        # paths in the metadata file look like 65/765.mp3
        relative_paths = [
            line.split("\t")[3][:-4].strip() for line in metadata[1:] if line
        ]
        if not self.split:
            # pretrain mode, so loading audio
            audio_format = self.feature_config["audio_format"]
            paths = [
                self.root / audio_format / f"{rel_path}.{audio_format}"
                for rel_path in relative_paths
            ]
        else:
            paths = [
                self.root / self.feature / f"{rel_path}.pt"
                for rel_path in relative_paths
            ]

        labels = [
            [tag.strip() for tag in line.split("\t")[5:]] for line in metadata[1:]
        ]

        # remove missing tracks
        with open(self.metadata_path.parent / "missing_tracks.txt", "r") as f:
            missing_tracks = f.readlines()
        missing_tracks = [track.strip() for track in missing_tracks]
        # get idx of tracks to remove
        idx_to_remove = [
            idx
            for idx, path in enumerate(paths)
            if str(Path(path.parent.stem) / path.stem) in missing_tracks
        ]
        # remove from paths and labels
        paths = [path for idx, path in enumerate(paths) if idx not in idx_to_remove]
        labels = [label for idx, label in enumerate(labels) if idx not in idx_to_remove]

        # encode labels
        encoded_labels = self.mlb.fit(labels)
        encoded_labels = self.mlb.transform(labels)

        return paths, encoded_labels

    def load_track(self, path) -> Tensor:
        if self.split:
            # if split is specified, dataset is used for downstream task, thus load features
            return torch.load(path, weights_only=True)
        else:
            # if split is None, dataset is used for pretraining, thus load audio
            waveform, original_sample_rate = torchaudio.load(path, normalize=True)
            if waveform.size(0) != 1:  # make mono if stereo (or more)
                waveform = waveform.mean(dim=0, keepdim=True)
            if original_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sample_rate, new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            return waveform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        path = self.paths[idx]
        label = self.labels[idx]

        track = self.load_track(path)

        return track, label
