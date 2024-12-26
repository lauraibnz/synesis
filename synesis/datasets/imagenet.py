from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageNet as TorchImageNet

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
        fv: Optional[str] = None,
        ratio: Optional[float] = 0.1,
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
        """
        self.feature = feature
        self.root = root
        if split not in [None, "train", "test", "validation"]:
            raise ValueError(
                f"Invalid split: {split}"
                + "Options: None, 'train', 'test', 'validation'"
            )
        self.item_format = item_format
        self.fv = fv
        self.ratio = ratio
        if download:
            raise NotImplementedError("Download not supported for ImageNet")

        # Load feature extractor config
        if not feature_config:
            feature_config = feature_configs[feature]
        self.feature_config = feature_config

        # Define transforms based on model requirements
        self.transform = T.Compose(
            [
                T.Resize(feature_config["input_size"]),
                T.CenterCrop(feature_config["input_size"]),
                T.ToTensor(),
                T.Normalize(mean=feature_config["mean"], std=feature_config["std"]),
            ]
        )

        # Initialize torchvision ImageNet
        split_map = {
            "train": "train",
            "test": "val",
            "validation": "train",
        }
        self.dataset = TorchImageNet(
            root=str(self.root),
            split=split_map[self.split] if self.split else None,
            transform=self.transform,
        )

        # Perform stratified split if necessary
        if self.split in ["train", "validation"]:
            self._stratified_split()

        # Reduce dataset size if ratio is provided
        if self.ratio is not None:
            self._reduce_dataset()

    def _stratified_split(self):
        indices = list(range(len(self.dataset)))
        labels = [self.dataset[i][1] for i in indices]

        train_indices, val_indices = train_test_split(
            indices, test_size=0.1, stratify=labels, random_state=self.seed
        )

        if self.split == "train":
            self.dataset = Subset(self.dataset, train_indices)
        elif self.split == "validation":
            self.dataset = Subset(self.dataset, val_indices)

    def _reduce_dataset(self):
        total_size = len(self.dataset)
        reduced_size = int(total_size * self.ratio)
        rng = torch.Generator().manual_seed(self.seed)
        reduced_indices = torch.randperm(total_size, generator=rng).tolist()[
            :reduced_size
        ]
        self.dataset = Subset(self.dataset, reduced_indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        if self.fv:
            # calculate fv
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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

        return image, label
