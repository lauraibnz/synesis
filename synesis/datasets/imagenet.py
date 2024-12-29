from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from config.features import configs as feature_configs


class ImageNet(Dataset):
    def __init__(
        self,
        feature: str,
        root: Union[str, Path] = "data/ImageNet",
        split: Optional[str] = None,
        download: bool = False,
        feature_config: Optional[dict] = None,
        item_format: str = "feature",
        image_format: str = "JPEG",
        fv: Optional[str] = None,
        ratio: Optional[float] = 0.1,
        itemization: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        """ImageNet dataset implementation supporting different feature extractors.

        Args:
            feature: Name of the feature extractor to use
            root: Root directory containing ImageNet data
            split: 'train', 'validation', 'test' or None
            download: Whether to download dataset
            feature_config: Override default feature extractor config
            fv: Factor of variation (i.e. label) to return
            ratio: Ratio for using a subset of the dataset
            item_format: 'raw' or 'feature'
            itemization: ignored, for compatibility with other datasets
        """
        self.feature = feature
        self.root = root
        if split not in [None, "train", "test", "validation"]:
            raise ValueError(
                f"Invalid split: {split}"
                + "Options: None, 'train', 'test', 'validation'"
            )
        self.split = split
        self.item_format = item_format
        self.fv = fv
        self.ratio = ratio
        self.image_format = image_format
        self.seed = seed
        self.label_encoder = LabelEncoder()
        if download:
            raise NotImplementedError("Download not supported for ImageNet")

        # Load feature extractor config
        if not feature_config:
            feature_config = feature_configs[feature]
        self.feature_config = feature_config

        # Define transforms based on model requirements
        self.transform_1 = T.Compose(
            [
                T.Resize(feature_config["resize_dim"]),
                T.CenterCrop(feature_config["input_dim"]),
            ]
        )
        self.transform_2 = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=feature_config.get("mean", [0.485, 0.456, 0.406]),
                    std=feature_config.get("std", [0.229, 0.224, 0.225]),
                ),
            ]
        )

        self._load_metadata()

        # Reduce dataset size if ratio is provided
        if self.ratio is not None:
            self._reduce_dataset()

    def _load_metadata(self):
        """Build dataset by scanning directory structure."""
        # Define split directory mapping
        split_map = {"train": "train", "validation": "train", "test": "val"}

        self.raw_data_paths = []
        self.labels = []

        if self.split is None:
            splits = ["train", "test"]  # no validation, as we're not splitting
        else:
            splits = [self.split]

        for split in splits:
            split_dir = (
                Path(self.root) / self.image_format / split_map.get(split, split)
            )
            if split == "test":
                metadata_path = Path(self.root) / "metadata" / "LOC_val_solution.csv"
                # read line by line
                with open(metadata_path, "r") as f:
                    f.readline()
                    for line in f:
                        img_name, label = line.strip().split(",")
                        label = label.split(" ")[0]  # remove bounding box info
                        self.raw_data_paths.append(
                            str(split_dir / f"{img_name}.{self.image_format}")
                        )
                        self.labels.append(label)
            else:
                # traverse subdirs
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        for img_path in class_dir.glob("*.JPEG"):
                            self.raw_data_paths.append(str(img_path))
                            self.labels.append(class_dir.name)

        labels = self.label_encoder.fit_transform(self.labels)
        self.labels = torch.tensor(labels, dtype=torch.long)

        # if train or validation, need to split the original train set
        if self.split in ["train", "validation"]:
            train_indices, val_indices = train_test_split(
                range(len(self.raw_data_paths)),
                test_size=0.1,
                stratify=self.labels,
                random_state=42,
            )
            if self.split == "train":
                self.raw_data_paths = [self.raw_data_paths[i] for i in train_indices]
                self.labels = self.labels[train_indices]
            elif self.split == "validation":
                self.raw_data_paths = [self.raw_data_paths[i] for i in val_indices]
                self.labels = self.labels[val_indices]

        self.feature_paths = [
            path.replace(f".{self.image_format}", ".pt")
            .replace(f"/{self.image_format}/", f"/{self.feature}/")
            .replace("/img/", f"/{self.feature}/")
            for path in self.raw_data_paths
        ]
        self.paths = (
            self.raw_data_paths if self.item_format == "raw" else self.feature_paths
        )

    def _reduce_dataset(self):
        """Reduce dataset size by sampling a subset based on a ratio."""
        num_samples = int(self.ratio * len(self))
        indices = np.random.choice(len(self), num_samples, replace=False)
        self.paths = [self.paths[i] for i in indices]
        self.labels = self.labels[indices]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        if self.item_format == "raw":
            image = Image.open(self.paths[idx]).convert("RGB")
            label = self.labels[idx]

            if self.transform_1:
                image = self.transform_1(image)

            if self.fv:
                image_np = np.array(image)
                hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
                hue, saturation, brightness = cv2.split(hsv_image)
                match self.fv:
                    case "hue":
                        label = np.mean(hue)
                    case "saturation":
                        label = np.mean(saturation)
                    case "brightness":
                        label = np.mean(brightness)
                    case "value":
                        label = np.mean(brightness)
                    case _:
                        raise ValueError(f"Invalid factor of variation: {self.fv}")

            if self.transform_2:
                image = self.transform_2(image)

        elif self.item_format == "feature":
            image = torch.load(self.paths[idx], weights_only=False)
            image = image.unsqueeze(0)
            label = self.labels[idx]

            if self.fv:
                # need to load image to compute
                image = Image.open(self.raw_data_paths[idx]).convert("RGB")
                image = self.transform_1(image)
                image_np = np.array(image)
                hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
                hue, saturation, brightness = cv2.split(hsv_image)
                match self.fv:
                    case "hue":
                        label = np.mean(hue)
                    case "saturation":
                        label = np.mean(saturation)
                    case "brightness":
                        label = np.mean(brightness)
                    case "value":
                        label = np.mean(brightness)
                    case _:
                        raise ValueError(f"Invalid factor of variation: {self.fv}")
        else:
            raise ValueError(f"Invalid item format: {self.item_format}")

        label = torch.tensor(label, dtype=torch.float32)

        return image, label
