from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from imgaug import augmenters as iaa
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from config.features import configs as feature_configs
from config.transforms import configs as transform_configs


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
        label: str = "class",
        transform: Optional[str] = None,
        ratio: Optional[float] = 0.005,
        itemization: Optional[str] = None,
        norm: bool = False,
        target_norm: bool = False,
        seed: int = 42,
    ) -> None:
        """ImageNet dataset implementation supporting different feature extractors.

        Args:
            feature: Name of the feature extractor to use
            root: Root directory containing ImageNet data
            split: 'train', 'validation', 'test' or None
            download: Whether to download dataset
            feature_config: Override default feature extractor config
            label: Factor of variation (i.e. label) to return
            transform: Transformation to apply to images, makes the dataset return
                       two items: the original and the transformed image
                       !NOTE hacky, need to refactor
            ratio: Ratio for using a subset of the dataset
            item_format: 'raw' or 'feature'
            itemization: ignored, for compatibility with other datasets
            norm: whether to convert to tensor and normalize before returning
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
        self.label = label
        self.transform = transform
        self.ratio = ratio
        self.image_format = image_format
        self.seed = seed
        self.norm = norm
        self.label_encoder = LabelEncoder()
        if download:
            raise NotImplementedError("Download not supported for ImageNet")

        # Load feature extractor config
        if not feature_config:
            feature_config = feature_configs[feature]
        self.feature_config = feature_config

        # Define transforms based on model requirements
        self.resize_and_crop = T.Compose(
            [
                T.Resize(feature_config["resize_dim"]),
                T.CenterCrop(feature_config["input_dim"]),
            ]
        )
        self.tensor_and_norm = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=feature_config.get("mean", [0.485, 0.456, 0.406]),
                    std=feature_config.get("std", [0.229, 0.224, 0.225]),
                ),
            ]
        )

        self._load_metadata()

    def _load_metadata(self):
        """Build dataset by scanning directory structure."""
        # Define split directory mapping
        split_map = {"train": "train", "validation": "train", "test": "val"}

        self.raw_data_paths = []
        self.labels = []

        if self.split is None:
            splits = ["train", "validation", "test"]
        else:
            splits = [self.split]

        # collect paths and labels for each split
        for split in splits:
            self.raw_data_paths.append([])
            self.labels.append([])
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
                        self.raw_data_paths[-1].append(
                            str(split_dir / f"{img_name}.{self.image_format}")
                        )
                        self.labels[-1].append(label)
            else:
                # traverse subdirs
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        for img_path in class_dir.glob("*.JPEG"):
                            self.raw_data_paths[-1].append(str(img_path))
                            self.labels[-1].append(class_dir.name)

        # if train or validation, need to split the original train set
        for i, split in enumerate(splits):
            if split in ["train", "validation"]:
                train_indices, val_indices = train_test_split(
                    range(len(self.raw_data_paths[i])),
                    test_size=0.1,
                    stratify=self.labels[i],
                    random_state=42,
                )
                if split == "train":
                    self.raw_data_paths[i] = [
                        self.raw_data_paths[i][j] for j in train_indices
                    ]
                    self.labels[i] = [self.labels[i][j] for j in train_indices]
                elif split == "validation":
                    self.raw_data_paths[i] = [
                        self.raw_data_paths[i][j] for j in val_indices
                    ]
                    self.labels[i] = [self.labels[i][j] for j in val_indices]

        # if ratio provided, need to get a subset of each split
        if self.ratio:
            for i in range(len(splits)):
                indices = self.create_subset(self.raw_data_paths[i])
                self.raw_data_paths[i] = [self.raw_data_paths[i][j] for j in indices]
                self.labels[i] = [self.labels[i][j] for j in indices]

        size = [len(paths) for paths in self.raw_data_paths]
        size = sum(size)

        # flatten lists
        self.raw_data_paths = [path for paths in self.raw_data_paths for path in paths]
        self.labels = [label for labels in self.labels for label in labels]

        size_2 = len(set(self.raw_data_paths))

        assert size == size_2

        # encode labels
        self.labels = self.label_encoder.fit_transform(self.labels)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        self.feature_paths = [
            path.replace(f".{self.image_format}", ".pt")
            .replace(f"/{self.image_format}/", f"/{self.feature}/")
            .replace("/img/", f"/{self.feature}/")
            for path in self.raw_data_paths
        ]
        self.paths = (
            self.raw_data_paths if self.item_format == "raw" else self.feature_paths
        )

    def create_subset(self, paths: list):
        """Reduce dataset size by sampling a subset based on a ratio."""
        np.random.seed(self.seed)
        num_samples = int(self.ratio * len(paths))
        indices = np.random.choice(len(paths), num_samples, replace=False)
        return indices

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        if self.item_format == "raw":
            image = Image.open(self.paths[idx]).convert("RGB")
            image = self.resize_and_crop(image)

            if self.transform:
                # get transform config
                transform_config = transform_configs.get(self.transform)
                if "HueShift" in self.transform:
                    tf_param = np.random.uniform(
                        transform_config["min"], transform_config["max"]
                    )
                    tf_image = TF.adjust_hue(image, tf_param)
                elif "SaturationShift" in self.transform:
                    tf_param = np.random.uniform(
                        transform_config["min"], transform_config["max"]
                    )
                    tf_image = TF.adjust_saturation(image, tf_param)
                elif "BrightnessShift" in self.transform:
                    tf_param = np.random.uniform(
                        transform_config["min"], transform_config["max"]
                    )
                    tf_image = TF.adjust_brightness(image, tf_param)
                elif "JPEGCompression" in self.transform:
                    tf_param = np.random.uniform(
                        transform_config["min"], transform_config["max"]
                    )
                    aug = iaa.JpegCompression(compression=tf_param)
                    tf_image = aug(image=np.array(image))
                # scale param to [0, 1] given range
                tf_param = (tf_param - transform_config["min"]) / (
                    transform_config["max"] - transform_config["min"]
                )

                image = self.tensor_and_norm(image)
                tf_image = self.tensor_and_norm(tf_image)
                image = torch.stack([image, tf_image])
                label = torch.tensor(tf_param, dtype=torch.float32)

                return image, label

            elif self.label:
                image_np = np.array(image)
                hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
                hue, saturation, brightness = cv2.split(hsv_image)
                match self.label:
                    case "hue":
                        label = np.mean(hue)
                    case "saturation":
                        label = np.mean(saturation)
                    case "brightness":
                        label = np.mean(brightness)
                    case "value":
                        label = np.mean(brightness)
                    case "class":
                        label = self.labels[idx]
                    case "dummy":
                        label = torch.tensor(0, dtype=torch.float32)
                    case _:
                        raise ValueError(f"Invalid label type: {self.label}")

                image = self.tensor_and_norm(image)

            else:
                image = self.tensor_and_norm(image)

        elif self.item_format == "feature":
            image = torch.load(self.paths[idx], weights_only=False)
            image = image.unsqueeze(0)

            if self.label:
                # need to load image to compute
                image = Image.open(self.raw_data_paths[idx]).convert("RGB")
                image = self.resize_and_crop(image)
                image_np = np.array(image)
                hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
                hue, saturation, brightness = cv2.split(hsv_image)
                match self.label:
                    case "hue":
                        label = np.mean(hue)
                    case "saturation":
                        label = np.mean(saturation)
                    case "brightness":
                        label = np.mean(brightness)
                    case "value":
                        label = np.mean(brightness)
                    case "class":
                        label = self.labels[idx]
                    case "dummy":
                        label = torch.tensor(0, dtype=torch.float32)
                    case _:
                        raise ValueError(f"Invalid factor of variation: {self.label}")
        else:
            raise ValueError(f"Invalid item format: {self.item_format}")

        label = torch.tensor(label, dtype=torch.float32)

        return image, label
