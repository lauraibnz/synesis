import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
from torch.utils.data import Dataset

from config.features import feature_configs
from synesis.datasets.dataset_utils import load_track
from synesis.utils import download_github_dir, download_github_file


class MTGJamendo(Dataset):
    def __init__(
        self,
        feature: str,
        root: Union[str, Path] = "data/MTGJamendo",
        subset: Optional[str] = None,
        split: Optional[str] = None,
        download: bool = False,
        feature_config: Optional[dict] = None,
        audio_format: str = "mp3",
        item_format: str = "feature",
        seed: int = 42,
    ) -> None:
        """
        MTG Jamendo (and subset) dataset implementation.

        Args:
            feature: If split is None, prepare dataset for this feature extractor.
                     If split is not None, load these extracted features.
            root: Root directory of the dataset. Defaults to "data/MTGJamendo".
            subset: Subset of the dataset to use: ["top50tags", "genre", "instrument",
                    "moodtheme", None], where None uses the full dataset.
            split: Split of the dataset to use: ["train", "test", "validation", None],
                     where None uses the full dataset (e.g. for feature extraction).
            download: Whether to download the dataset if it doesn't exist.
            feature_config: Configuration for the feature extractor.
            audio_format: Format of the audio files: ["mp3", "wav", "ogg"].
            item_format: Format of the items to return: ["audio", "feature"].
            seed: Random seed for reproducibility.
        """
        self.tasks = ["tagging"]
        self.fvs = ["key", "tempo", "eq"]

        root = Path(root)
        self.root = root
        if subset not in [None, "top50tags", "genre", "instrument", "moodtheme"]:
            raise ValueError(
                f"Invalid subset: {subset} "
                + "Options: None, 'top50tags', 'genre', 'instrument', 'moodtheme'"
            )
        if split not in [None, "train", "test", "validation"]:
            raise ValueError(
                f"Invalid split: {split} "
                + "Options: None, 'train', 'test', 'validation'"
            )
        self.subset = subset
        self.split = split
        self.feature = feature
        self.item_format = item_format
        self.audio_format = audio_format

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
            if self.subset:
                self.metadata_path = (
                    root / "metadata" / f"autotagging_{subset}-{split}.tsv"
                )
            else:
                self.metadata_path = root / "metadata" / f"autotagging-{split}.tsv"

        if not feature_config:
            # load default feature config
            feature_config = feature_configs[feature]
        self.feature_config = feature_config

        self.label_encoder = MultiLabelBinarizer()

        self._load_metadata()

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
            "autotagging_top50tags.tsv",
            "raw_30s_cleantags_50artists.tsv",
        ]:
            download_github_file(
                owner="MTG",
                repo="mtg-jamendo-dataset",
                file_path=f"data/{f}",
                save_dir=str(self.root / "metadata"),
            )

    def _load_metadata(self) -> Tuple[list, torch.Tensor]:
        with open(self.metadata_path, "r") as f:
            metadata = f.readlines()
        # paths in the metadata file look like 65/765.mp3
        relative_paths = [
            line.split("\t")[3][:-4].strip() for line in metadata[1:] if line
        ]
        if self.item_format == "audio":
            paths = [
                self.root / self.audio_format / f"{rel_path}.{self.audio_format}"
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
        encoded_labels = self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)

        self.audio_paths, self.labels = paths, encoded_labels

        self.feature_paths = [
            str(path)
            .replace(f".{self.audio_format}", ".pt")
            .replace("mp3", self.feature)
            for path in paths
        ]
        self.paths = (
            self.audio_paths if self.item_format == "audio" else self.feature_paths
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        path = (
            self.audio_paths[idx]
            if self.item_format == "audio"
            else self.feature_paths[idx]
        )
        label = self.labels[idx]

        track = load_track(
            path=path,
            item_format=self.item_format,
            sample_rate=self.feature_config["sample_rate"],
        )

        return track, label
