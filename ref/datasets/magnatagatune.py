import csv
import os.path
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import wget
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from config.features import feature_config as fc


class MagnaTagATune(Dataset):
    def __init__(
        self,
        feature: str,
        root: Union[str, Path] = "data/MagnaTagATune",
        split: Optional[str] = None,
        download: bool = False,
        feature_config: Optional[dict] = None,
        audio_format: str = "mp3",
        item_format: str = "feature",
        seed: int = 42,
    ) -> None:
        """MagnaTagATune dataset implementation.

        Args:
            feature: If split is None, prepare dataset for this feature extractor.
                     If split is not None, load these extracted features.
            root: Root directory of the dataset. Defaults to "data/MagnaTagATune".
            split: Split of the dataset to use: ["train", "test", "val", None], where
                     None uses the full dataset (e.g. for feature extraction).
            download: Whether to download the dataset if it doesn't exist.
            feature_config: Configuration for the feature extractor.
            audio_format: Format of the audio files: ["mp3", "wav", "ogg"].
            item_format: Format of the items to return: ["audio", "feature"].
            seed: Random seed for reproducibility.
        """
        self.tasks = ["tagging"]

        root = Path(root)
        self.root = root
        if split not in [None, "train", "test", "val"]:
            raise ValueError(
                f"Invalid split: {split} " + "Options: None, 'train', 'test', 'val'"
            )
        self.split = split
        self.item_format = item_format
        self.audio_format = audio_format
        self.feature = feature
        self.mlb = MultiLabelBinarizer()

        if not feature_config:
            # load default feature config
            feature_config = fc[feature]
        self.feature_config = feature_config

        if download:
            self._download()

        self.paths, self.labels = self._load_metadata()

    def _download(self) -> None:
        # make data dir if it doesn't exist or if it exists but is empty
        if os.path.exists(os.path.join(self.root, "mp3")) and (
            len(os.listdir(os.path.join(self.root, "mp3"))) != 0
        ):
            import warnings

            warnings.warn(
                f"Dataset '{self.name}' already exists in '{self.root}'."
                + "Skipping audio download.",
                stacklevel=2,
            )
            self.download_metadata()
            return
        (Path(self.root) / "mp3").mkdir(parents=True, exist_ok=True)

        print(f"Downloading MagnaTagATune to {self.root}...")
        for i in tqdm(["001", "002", "003"]):
            wget.download(
                url=f"https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.{i}",
                out=str(Path(self.root) / "mp3"),
            )

        archive_dir = Path(self.root) / "mp3"
        # Combine the split archive files into a single file
        with open(archive_dir / "mp3.zip", "wb") as f:
            for i in ["001", "002", "003"]:
                with open(
                    os.path.join(archive_dir, f"mp3.zip.{i}"),
                    "rb",
                ) as part:
                    f.write(part.read())

        # Extract the contents of the archive
        with zipfile.ZipFile(archive_dir / "mp3.zip", "r") as zip_ref:
            zip_ref.extractall(path=archive_dir)

        # Remove zips
        for i in ["", ".001", ".002", ".003"]:
            os.remove(os.path.join(archive_dir, f"mp3.zip{i}"))

        # Download metadata
        if os.path.exists(os.path.join(self.root, "metadata")) and (
            len(os.listdir(os.path.join(self.root, "metadata"))) != 0
        ):
            import warnings

            warnings.warn(
                f"Metadata for dataset '{self.name}' already exists in '{self.root}'."
                + "Skipping metadata download.",
                stacklevel=2,
            )
            return
        (Path(self.root) / "metadata").mkdir(parents=True, exist_ok=True)

        urls = [
            # annotations
            "https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv",
            # train, validation, and test splits from Won et al. 2020
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/train.npy",
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/valid.npy",
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/test.npy",
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/tags.npy",
        ]
        for url in urls:
            wget.download(
                url=url,
                out=os.path.join(self.root, "metadata/"),
            )

    def _load_metadata(self) -> Tuple[list, np.ndarray]:
        # load track ids
        with open(
            os.path.join(self.root, "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            next(annotations)  # skip header
            self.track_ids = [
                line[0]
                for line in annotations
                if line[0] not in ["35644", "55753", "57881"]
            ]

        # load audio paths
        with open(
            os.path.join(self.root, "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            next(annotations)  # skip header
            self.audio_paths = [
                os.path.join(self.root, "mp3", line[-1])
                for line in annotations
                # remove some corrupted files
                if line[0] not in ["35644", "55753", "57881"]
            ]

        # load labels
        # get the list of top 50 tags used in Minz Won et al. 2020
        tags = np.load(os.path.join(self.root, "metadata", "tags.npy"))

        with open(
            os.path.join(self.root, "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            annotations_header = next(annotations)
            labels = [
                [
                    annotations_header[j]
                    for j in range(1, len(line) - 1)
                    # only add the tag if it's in the tags list
                    if line[j] == "1" and annotations_header[j] in tags
                ]
                for line in annotations
                # remove some corrupted files
                if line[0] not in ["35644", "55753", "57881"]
            ]
        # encode labels
        encoded_labels = self.mlb.fit(labels)
        encoded_labels = self.mlb.transform(labels)

        labels = np.array(encoded_labels)

        if self.item_format == "audio":
            return self.audio_paths, labels

        # load splits
        if self.split:
            relative_paths_in_split = np.load(
                os.path.join(self.root, "metadata", f"{self.split}.npy")
            )
            # clean up
            relative_paths_in_split = [
                rp.split("\t")[1] for rp in relative_paths_in_split
            ]

            # get the indices of the tracks in the split
            indices = [
                i
                for i, path in enumerate(self.audio_paths)
                if path in relative_paths_in_split
            ]

            # keep these indices in path and labels
            self.audio_paths = [self.audio_paths[i] for i in indices]
            labels = labels[indices]

        feature_paths = [
            path.replace(f".{self.audio_format}", ".pt").replace("mp3", self.feature)
            for path in self.audio_paths
        ]

        return feature_paths, labels

    def load_track(self, path) -> Tensor:
        if self.item_format == "feature":
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

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        path = self.paths[idx]
        label = self.labels[idx]

        track = self.load_track(path)

        return track, label
