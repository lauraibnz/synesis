import csv
import os
import os.path
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
import torch
import wget
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from config.features import configs as feature_configs
from synesis.datasets.dataset_utils import load_track


class SynTheory(Dataset):
    def __init__(
        self,
        feature: str,
        root: Union[str, Path] = "data/SynTheory",
        split: Optional[str] = None,
        download: bool = False,
        feature_config: Optional[dict] = None,
        audio_format: str = "wav",
        item_format: str = "feature",
        itemization: bool = True,
        seed: int = 42,
        label: str = "notes",  # Mandatory - the concept to probe
        transform=None,  # NOTE: ignored, for compatibility
    ) -> None:
        """SynTheory dataset implementation.

        Args:
            feature: If split is None, prepare dataset for this feature extractor.
                     If split is not None, load these extracted features.
            root: Root directory of the dataset. Defaults to "data/SynTheory".
            split: Split of the dataset to use: ["train", "test", "validation", None],
                     where None uses the full dataset (e.g. for feature extraction).
            download: Whether to download the dataset if it doesn't exist.
            feature_config: Configuration for the feature extractor.
            audio_format: Format of the audio files: ["wav", "mp3", "ogg"].
            item_format: Format of the items to return: ["raw", "feature"].
            itemization: For datasets with variable-length items, whether to return them
                      as a list of equal-length items (True) or as a single item.
            seed: Random seed for reproducibility.
            label: The music theory concept to predict: ["notes", "tempos", "time_signatures", etc.]
            transform: Ignored, for compatibility with other datasets.
        """
        self.tasks = ["classification", "regression"]
        
        # Concept labels mapping: (num_classes, column_name)
        # We take the first (most important) label for each concept
        self.concept_labels = {
            "chord_progressions": (19, "chord_progression"),
            "chords": (4, "chord_type"), 
            "scales": (7, "mode"),
            "intervals": (12, "interval"),
            "notes": (12, "root_note_pitch_class"),
            "time_signatures": (8, "time_signature"),
            "tempos": (161, "bpm"),
            "instruments": (92, "midi_program_num"), 
            "categories": (13, "midi_category"),
        }
    
        self.concepts = list(self.concept_labels.keys())

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
        self.seed = seed
        self.label = label  # This is our concept to probe
        
        # Validate concept
        if self.label not in self.concepts:
            raise ValueError(
                f"Unknown concept: {self.label}. "
                f"Available concepts: {self.concepts}"
            )
        
        # Get the main label column and number of classes for this concept
        self.n_classes, self.target_column = self.concept_labels[self.label]
        
        # Set up task type and label encoder
        if self.label == "tempos":
            # Regression task
            self.is_regression = True
            self.label_encoder = StandardScaler()
        else:
            # Classification task
            self.is_regression = False
            self.label_encoder = LabelEncoder()

        if not feature_config:
            # load default feature config
            feature_config = feature_configs[feature]
        self.feature_config = feature_config

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata and create train/test/validation splits."""
        # Load the info.csv file for this specific concept
        # Special handling for instruments which uses the notes dataset
        if self.label == "instruments" or self.label == "categories":
            dataset_dir = "chords"
        else:
            dataset_dir = self.label
            
        info_path = self.root / dataset_dir / "info.csv"
        if not info_path.exists():
            raise FileNotFoundError(f"Info file not found: {info_path}")
        
        # Load the dataset info
        df = pd.read_csv(info_path)
        
        # Ensure we have the required columns
        required_cols = ["synth_file_path", self.target_column]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in info.csv")
        
        # Create full audio paths
        audio_paths = []
        for audio_path in df["synth_file_path"]:
            full_path = self.root / dataset_dir / audio_path
            audio_paths.append(str(full_path))
        
        # Get labels from the target column
        labels = df[self.target_column].values
        
        # Handle regression vs classification
        if self.is_regression:
            # For regression (e.g., tempos), normalize to [0, 1]
            if hasattr(self.label_encoder, 'fit_transform'):
                labels = self.label_encoder.fit_transform(labels.reshape(-1, 1)).flatten()
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            # For classification, encode labels
            encoded_labels = self.label_encoder.fit_transform(labels)
            labels = torch.tensor(encoded_labels, dtype=torch.long)
        # Create train/test/validation splits
        if self.split:
            np.random.seed(self.seed)
            indices = np.arange(len(audio_paths))
            np.random.shuffle(indices)
            
            # Special handling for tempos (out-of-domain split)
            if self.label == "tempos" and self.is_regression:
                # Sort by tempo for out-of-domain split
                tempo_values = df[self.target_column].values
                sorted_indices = np.argsort(tempo_values)
                
                # Use middle 70% for training, outer 30% for test/val
                n_samples = len(sorted_indices)
                train_start = int(0.15 * n_samples)
                train_end = int(0.85 * n_samples)
                
                train_indices = sorted_indices[train_start:train_end]
                holdout_indices = np.concatenate([
                    sorted_indices[:train_start], 
                    sorted_indices[train_end:]
                ])
                
                # Split holdout into test and validation
                np.random.shuffle(holdout_indices)
                mid_point = len(holdout_indices) // 2
                test_indices = holdout_indices[:mid_point]
                val_indices = holdout_indices[mid_point:]
                
            else:
                # Standard random split: 70% train, 15% test, 15% validation
                n_samples = len(indices)
                train_end = int(0.7 * n_samples)
                test_end = int(0.85 * n_samples)
                
                train_indices = indices[:train_end]
                test_indices = indices[train_end:test_end]
                val_indices = indices[test_end:]
            
            # Select indices based on split
            if self.split == "train":
                selected_indices = train_indices
            elif self.split == "test":
                selected_indices = test_indices
            elif self.split == "validation":
                selected_indices = val_indices
            
            # Filter data based on selected indices
            audio_paths = [audio_paths[i] for i in selected_indices]
            labels = labels[selected_indices]
        
        self.raw_data_paths = audio_paths
        self.labels = labels
        
        # Create feature paths
        self.feature_paths = []
        for path in audio_paths:
            split_path = os.path.split(path)
            # Replace file extension properly: file.wav -> file.pt
            filename = os.path.splitext(split_path[1])[0] + ".pt"
            # Use the label for feature path (not dataset_dir) to keep features organized by concept
            feature_path = os.path.join(split_path[0], self.feature, filename)
            self.feature_paths.append(feature_path)
        
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
