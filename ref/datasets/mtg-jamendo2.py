from itertools import chain
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
from torch.utils.data import Dataset

from config.features import feature_config as fc


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
        self.subset = subset
        if self.subset:
            self.metadata_path = root / "metadata" / f"autotagging_{subset}.tsv"
        else:
            self.metadata_path = root / "metadata" / "raw_30s_cleantags_50artists.tsv"

        if not feature_config:
            feature_config = fc[feature]

        self.mlb = MultiLabelBinarizer()

        self.paths, self.labels = self._load_metadata()
        if split:
            self.paths, self.labels = self._split(split)

    def _load_metadata(self):
        with open(self.metadata_path, "r") as f:
            metadata = f.readlines()
        # paths in the metadata file look like 65/765.mp3
        relative_paths = [
            line.split("\t")[3][:-4].strip() for line in metadata[1:] if line
        ]
        if self.format == "melspec":
            paths = [
                self.root / "melspec" / f"{rel_path}.pt" for rel_path in relative_paths
            ]
        else:
            paths = [
                self.root / self.format / f"{rel_path}.{self.format}"
                for rel_path in relative_paths
            ]

        labels = [
            [tag.strip() for tag in line.split("\t")[5:]] for line in metadata[1:]
        ]

        # only keep tracks that contain at least one of the top 100 tags
        all_tags = set(tag for tags in labels for tag in tags)
        tag_counts = {tag: 0 for tag in all_tags}
        for tags in labels:
            for tag in tags:
                tag_counts[tag] += 1
        top_100_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:100]
        self.top_100_tags = top_100_tags

        track_indices_to_keep = [
            idx
            for idx, tags in enumerate(labels)
            if any(tag in top_100_tags for tag in tags)
        ]

        paths = [paths[idx] for idx in track_indices_to_keep]
        labels = [labels[idx] for idx in track_indices_to_keep]

        # remove non top 100 tags
        for idx, tags in enumerate(labels):
            labels[idx] = [tag for tag in tags if tag in top_100_tags]

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

    def _split(self):
        np.random.seed(self.random_seed)

        if self.train_size > 1:
            self.train_size = self.train_size / len(self.paths)
        X_train, X_test, y_train, y_test = self._multilabel_train_test_split(
            self.paths,
            self.labels,
            stratify=self.labels,
            train_size=self.train_size,
        )
        if self.train is False:
            return X_test, y_test

        return X_train, y_train

    def _multilabel_train_test_split(self, *arrays, train_size, stratify=None):
        """
        Train test split for multilabel classification. Uses the
        algorithm from: 'Sechidis K., Tsoumakas G., Vlahavas I.
        (2011) On the Stratification of Multi-Label Data'.
        """
        assert stratify is not None

        arrays = indexable(*arrays)
        n_samples = _num_samples(arrays[0])
        n_train, n_test = _validate_shuffle_split(n_samples, None, train_size)
        cv = MultilabelStratifiedShuffleSplit(
            test_size=n_test, train_size=n_train, random_state=self.random_seed
        )
        train, test = next(cv.split(X=arrays[0], y=stratify))

        return list(
            chain.from_iterable(
                (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
            )
        )

    def load_track(self, path) -> Tensor:
        if self.format == "melspec":
            return torch.load(path, weights_only=True)
        else:
            waveform, original_sample_rate = torchaudio.load(path, normalize=True)
            if waveform.size(0) != 1:  # make mono if stereo (or more)
                waveform = waveform.mean(dim=0, keepdim=True)
            if original_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sample_rate, new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            return waveform

    def sample_items(self, track: Tensor) -> Tensor:
        """Sample a fixed number of items from a track.
        Subsample equally spaced items if track is longer,
        else take overlapping items, also equally spaced
        (both cases are essentially handled the same way,
        by creating an equally spaced grid of item starts).
        """
        if self.format == "melspec":
            # Get melspec item length in samples: floor((N - W) / H) + 1
            item_len = (
                int(
                    (self.item_len_sec * self.sample_rate - self.mel_fft_size)
                    / self.mel_hop_size
                )
                + 1
            )
        else:
            # Get waveform item length in samples
            item_len = int(self.item_len_sec * self.sample_rate)

        item_starts = torch.linspace(
            0, track.size(1) - item_len, self.items_per_track
        ).int()
        items = torch.stack(
            [track[:, start : start + item_len] for start in item_starts]
        )

        return items

    def __len__(self) -> int:
        return len(self.paths * self.items_per_track)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """The index refers to the item. However, we're going to get the
        track index as the divisor of the index by the number of items per
        track,  and the item index (from the item sampling function) from
        the remainder of the division.
        """
        track_idx = idx // self.items_per_track
        item_idx = idx % self.items_per_track

        if not self.feature:
            path = self.paths[track_idx]
            track = self.load_track(path, weights_only=True)
            # welp it's currently sized as [2, 128, time] cos I forgot to
            # toggle mono while computing them, so I have to mean average
            track = track.mean(dim=0, keepdim=False)
            item = self.sample_items(track)[item_idx]
        else:
            # items are already computed
            path = str(self.paths[track_idx]).replace(self.format, self.feature)
            item = torch.load(path, weights_only=True)[item_idx]
        label = self.labels[track_idx]

        return item, label
