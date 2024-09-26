import csv
import os.path
import warnings
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import wget
from torch.utils.data import Dataset
from tqdm import tqdm


class MagnaTagATune(Dataset):
    def __init__(
        self,
        feature: str,
        root: Union[str, Path] = "data/MagnaTagATune",
        split: Optional[str] = None,
        download: bool = False,
        feature_config: Optional[dict] = None,
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
            seed: Random seed for reproducibility.
        """
        root = Path(root)
        self.root = root
        if split not in [None, "train", "test", "val"]:
            raise ValueError(
                f"Invalid split: {split} " + "Options: None, 'train', 'test', 'val'"
            )
        self.split = split

        if download:
            self._download()

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
        (Path(self.root) / "audio").mkdir(parents=True, exist_ok=True)

        print(f"Downloading MagnaTagATune to {self.root}...")
        for i in tqdm(["001", "002", "003"]):
            wget.download(
                url=f"https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.{i}",
                out=os.path.join(self.root, "audio/"),
            )

        archive_dir = os.path.join(self.root, "mp3")

        # Combine the split archive files into a single file
        with open(os.path.join(archive_dir, "mp3.zip"), "wb") as f:
            for i in ["001", "002", "003"]:
                with open(
                    os.path.join(archive_dir, f"mp3.zip.{i}"),
                    "rb",
                ) as part:
                    f.write(part.read())

        # Extract the contents of the archive
        with zipfile.ZipFile(os.path.join(archive_dir, "mp3.zip"), "r") as zip_ref:
            zip_ref.extractall()

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
            os.path.join(self.data_dir, "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            next(annotations)  # skip header
            self.track_ids = [
                line[0] for line in annotations
                if line[0] not in ["35644", "55753", "57881"]
            ]

        # load audio paths
        with open(
            os.path.join(self.root, "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            next(annotations)  # skip header
            self.audio_paths = [
                os.path.join(self.root, "audio", line[-1])
                for line in annotations
                # remove some corrupted files
                if line[0] not in ["35644", "55753", "57881"]
            ]

        # get the list of top 50 tags used in Minz Won et al. 2020
        tags = np.load(os.path.join(self.root, "metadata", "tags.npy"))

        with open(
            os.path.join(self.root, "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            annotations_header = next(annotations)
            self.labels = [[
                    annotations_header[j]
                    for j in range(1, len(line) - 1)
                    # only add the tag if it's in the tags list
                    if line[j] == "1" and annotations_header[j] in tags
                ]
                for line in annotations
                # remove some corrupted files
                if line[0] not in ["35644", "55753", "57881"]
            ]
            self.labels = np.array(self.labels)

        if not self.split:
            return self.audio_paths, self.labels
