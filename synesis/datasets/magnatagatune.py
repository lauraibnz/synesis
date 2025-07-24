import csv
import os.path
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import wget
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
import json

from config.features import configs as feature_configs
from synesis.datasets.dataset_utils import load_track


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
        itemization: bool = True,
        seed: int = 42,
        label: str = "",
        transform=None,  # !NOTE ignored, for compatability
    ) -> None:
        """MagnaTagATune dataset implementation.

        Args:
            feature: If split is None, prepare dataset for this feature extractor.
                     If split is not None, load these extracted features.
            root: Root directory of the dataset. Defaults to "data/MagnaTagATune".
            split: Split of the dataset to use: ["train", "test", "validation", None],
                     where None uses the full dataset (e.g. for feature extraction).
            download: Whether to download the dataset if it doesn't exist.
            feature_config: Configuration for the feature extractor.
            audio_format: Format of the audio files: ["mp3", "wav", "ogg"].
            item_format: Format of the items to return: ["raw", "feature"].
            itemization: For datasets with variable-length items, whether to return them
                      as a list of equal-length items (True) or as a single item.
            seed: Random seed for reproducibility.
        """
        self.tasks = ["tagging"]
        self.fvs = ["pitch", "tempo", "eq"]

        root = Path(root)
        self.root = root
        if split not in [None, "train", "test", "validation"]:
            raise ValueError(
                f"Invalid split: {split} "
                + "Options: None, 'train', 'test', 'validation'"
            )
        self.split = split
        self.item_format = item_format
        self.itemization = itemization
        self.audio_format = audio_format
        self.feature = feature
        self.label_encoder = MultiLabelBinarizer()
        self.categories = self._load_categories()
        self.label = label

        if not feature_config:
            # load default feature config
            feature_config = feature_configs[feature]
        self.feature_config = feature_config
        if download:
            self._download()

        self._load_metadata()

    def _download(self) -> None:
        # make data dir if it doesn't exist or if it exists but is empty
        if os.path.exists(os.path.join(self.root, "mp3")) and (
            len(os.listdir(os.path.join(self.root, "mp3"))) != 0
        ):
            import warnings

            warnings.warn(
                f"Dataset '{self.__class__.__name__}' already exists in '{self.root}'."
                + "Skipping audio download.",
                stacklevel=2,
            )
        else:
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
                f"Metadata for dataset '{self.__class__.__name__}' already exists in '{self.root}'."
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

    def _load_categories(self) -> dict:
        """Load categories from a JSON file."""
        categories_path = os.path.join(self.root, "metadata/categories.json")
        if os.path.exists(categories_path):
            with open(categories_path, "r") as f:
                return json.load(f)
        else:
            return {}

    def _load_metadata(self) -> Tuple[list, torch.Tensor]:
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
            raw_data_paths = [
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

        # Filter tags and corresponding labels based on the provided label
        if self.label in self.categories:
            category_tags = self.categories[self.label]
            tags = [tag for tag in tags if tag in category_tags]
            labels = [
                [tag for tag in label if tag in category_tags]
                for label in labels
            ]

        # encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)

        # load splits
        if self.split:
            relative_paths_in_split = np.load(
                os.path.join(
                    self.root,
                    "metadata",
                    "valid.npy" if self.split == "validation" else f"{self.split}.npy",
                )
            )
            # clean up
            relative_paths_in_split = [
                rp.split("\t")[1] for rp in relative_paths_in_split
            ]

            # get the indices of the tracks in the split
            indices = [
                i
                for i, path in enumerate(raw_data_paths)
                # I just want the name and its parent :')
                if str(Path(Path(path).parent.name) / Path(path).name)
                in relative_paths_in_split
            ]

            # keep these indices in path and labels
            raw_data_paths = [raw_data_paths[i] for i in indices]
            encoded_labels = encoded_labels[indices]

        self.raw_data_paths, self.labels = raw_data_paths, encoded_labels

        self.feature_paths = [
            str(path)
            .replace(f".{self.audio_format}", ".pt")
            .replace("mp3", self.feature)
            for path in raw_data_paths
        ]
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
        label = self.labels[idx]

        item_len_sec = self.feature_config.get("item_len_sec", None)

        track = load_track(
            path=path,
            item_format=self.item_format,
            itemization=self.itemization,
            item_len_sec=item_len_sec,
            sample_rate=self.feature_config["sample_rate"],
        )

        return track, label
