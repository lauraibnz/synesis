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
        transform: Optional[str] = None,  # Transform to apply (e.g., "InstrumentShift")
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
        self.transform = transform  # Store the transform
        
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
        
        # Store the full dataframe for finding similar samples when transforms are applied
        self.metadata_df = df
        
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
        # Always create splits (like other datasets) - during feature extraction or loading
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
            # Ensure at least 2 samples with same content but different instruments per split
            # For notes, chords, and instruments (which all have multiple instruments per content)
            if self.label in ["notes", "chords", "instruments"]:
                if self.label == "notes":
                    # Group by note + octave + register
                    df['content_key'] = df['root_note_pitch_class'].astype(str) + '_' + df['octave'].astype(str) + '_' + df['register'].astype(str)
                elif self.label == "chords" or self.label == "instruments":
                    # Group by chord type + root note
                    df['content_key'] = df['chord_type'].astype(str) + '_' + df['root_note_pitch_class'].astype(str)
                
                # Group samples by content key
                content_groups = df.groupby('content_key')
                
                # For each content group, ensure at least 2 different instruments are in the same split
                train_indices = []
                test_indices = []
                val_indices = []
                
                multi_instrument_groups = 0
                single_instrument_groups = 0
                
                for content_key, group in content_groups:
                    group_indices = group.index.tolist()
                    
                    # If this content has multiple instruments, ensure at least 2 in each split
                    if len(group) >= 6:  # Need at least 6 to put 2 in each split
                        multi_instrument_groups += 1
                        
                        # Shuffle the instruments for this content
                        np.random.shuffle(group_indices)
                        
                        # Put at least 2 instruments in each split
                        train_indices.extend(group_indices[:2])
                        test_indices.extend(group_indices[2:4])
                        val_indices.extend(group_indices[4:6])
                        
                        # Distribute remaining instruments randomly
                        remaining_indices = group_indices[6:]
                        for remaining_idx in remaining_indices:
                            split_choice = np.random.choice(['train', 'test', 'validation'], p=[0.7, 0.15, 0.15])
                            if split_choice == 'train':
                                train_indices.append(remaining_idx)
                            elif split_choice == 'test':
                                test_indices.append(remaining_idx)
                            else:  # validation
                                val_indices.append(remaining_idx)
                        

                    elif len(group) >= 2:
                        # For content with 2-5 instruments, distribute to ensure each split gets some
                        multi_instrument_groups += 1
                        np.random.shuffle(group_indices)
                        
                        # Distribute as evenly as possible
                        n_instruments = len(group_indices)
                        if n_instruments == 2:
                            train_indices.extend(group_indices[:1])
                            test_indices.extend(group_indices[1:2])
                        elif n_instruments == 3:
                            train_indices.extend(group_indices[:1])
                            test_indices.extend(group_indices[1:2])
                            val_indices.extend(group_indices[2:3])
                        elif n_instruments == 4:
                            train_indices.extend(group_indices[:2])
                            test_indices.extend(group_indices[2:3])
                            val_indices.extend(group_indices[3:4])
                        elif n_instruments == 5:
                            train_indices.extend(group_indices[:2])
                            test_indices.extend(group_indices[2:3])
                            val_indices.extend(group_indices[3:5])
                        

                    else:
                        single_instrument_groups += 1
                        # Single instrument content - assign randomly
                        split_choice = np.random.choice(['train', 'test', 'validation'], p=[0.7, 0.15, 0.15])
                        
                        if split_choice == 'train':
                            train_indices.extend(group_indices)
                        elif split_choice == 'test':
                            test_indices.extend(group_indices)
                        else:  # validation
                            val_indices.extend(group_indices)
                

                
            else:
                # Standard random split for other concepts
                n_samples = len(indices)
                train_end = int(0.7 * n_samples)
                test_end = int(0.85 * n_samples)
                
                train_indices = indices[:train_end]
                test_indices = indices[train_end:test_end]
                val_indices = indices[test_end:]
        
        # Store split indices for transform lookups
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.val_indices = val_indices
        
        # Select indices based on split (if specified)
        if self.split:
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

    def find_instrument_shifted_sample(self, idx: int) -> int:
        """Find a sample with the same musical content but different instrument."""
        # Get current sample info from the full dataset
        if self.split:
            # Map idx (within split) to full_idx (in full dataset)
            if self.split == "train":
                full_idx = self.train_indices[idx]
            elif self.split == "test":
                full_idx = self.test_indices[idx]
            elif self.split == "validation":
                full_idx = self.val_indices[idx]
        else:
            full_idx = idx
            
        current_row = self.metadata_df.iloc[full_idx]
        current_instrument = current_row['midi_program_num']
        
        # Get the current split indices
        if self.split:
            if self.split == "train":
                split_indices = self.train_indices
            elif self.split == "test":
                split_indices = self.test_indices
            elif self.split == "validation":
                split_indices = self.val_indices
        else:
            # No split, use all indices
            split_indices = np.arange(len(self.metadata_df))
        
        # Create a mask for the current split
        split_mask = np.isin(self.metadata_df.index, split_indices)
        current_split_df = self.metadata_df[split_mask]
        
        # Find samples with different instruments but same other factors within the current split
        # For notes dataset, we want same note, octave, register but different instrument
        if self.label == "notes":
            mask = (
                (current_split_df['root_note_pitch_class'] == current_row['root_note_pitch_class']) &
                (current_split_df['octave'] == current_row['octave']) &
                (current_split_df['register'] == current_row['register']) &
                (current_split_df['midi_program_num'] != current_instrument)
            )
        # For chords dataset, we want same chord type, root note but different instrument
        elif self.label == "chords" or self.label == "instruments":
            mask = (
                (current_split_df['root_note_pitch_class'] == current_row['root_note_pitch_class']) &
                (current_split_df['chord_type'] == current_row['chord_type']) &
                (current_split_df['midi_program_num'] != current_instrument)
            )
        # For other concepts, just find different instrument
        else:
            mask = (current_split_df['midi_program_num'] != current_instrument)
        
        # Get indices of matching samples within the split
        matching_indices = current_split_df[mask].index.tolist()
        
        if not matching_indices:
            # If no exact match, just find any different instrument within the split
            mask = (current_split_df['midi_program_num'] != current_instrument)
            matching_indices = current_split_df[mask].index.tolist()
        
        if not matching_indices:
            # If still no match, return the same index (fallback)
            return idx
        
        # Randomly select one of the matching samples (without affecting global seed)
        rng = np.random.RandomState()
        selected_idx = rng.choice(matching_indices)
        
        # Map the selected index back to the position within the current split
        mapped_idx = np.where(split_indices == selected_idx)[0][0]
        
        return mapped_idx

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # Load original sample
        path = (
            self.raw_data_paths[idx]
            if self.item_format == "raw"
            else self.feature_paths[idx]
        )
        label = self.labels[idx]

        item_len_sec = self.feature_config.get("item_len_sec", None)
        original_track = load_track(
            path=path,
            item_format=self.item_format,
            itemization=self.itemization,
            item_len_sec=item_len_sec,
            sample_rate=self.feature_config["sample_rate"],
        )

        # Apply transform if specified
        if self.transform == "InstrumentShift":
            # Find a sample with different instrument
            shifted_idx = self.find_instrument_shifted_sample(idx)
            

            
            # Load the shifted sample (could be same if no different instrument found)
            shifted_path = (
                self.raw_data_paths[shifted_idx]
                if self.item_format == "raw"
                else self.feature_paths[shifted_idx]
            )
            shifted_track = load_track(
                path=shifted_path,
                item_format=self.item_format,
                itemization=self.itemization,
                item_len_sec=item_len_sec,
                sample_rate=self.feature_config["sample_rate"],
            )
            
            # Determine which label to return based on label/transform relationship
            if self.label == "instruments":
                # For equivariance: return the transform (original -> transformed)
                original_label = self.labels[idx]
                shifted_label = self.labels[shifted_idx]
                
                # Create one-hot encodings for both original and transformed
                original_one_hot = torch.zeros(92, dtype=torch.float32)
                original_one_hot[original_label] = 1.0
                
                shifted_one_hot = torch.zeros(92, dtype=torch.float32)
                shifted_one_hot[shifted_label] = 1.0
                
                transform_param = shifted_one_hot - original_one_hot
                
                track = torch.stack([original_track, shifted_track], dim=0)
                return track, transform_param
            else:
                # For disentanglement: return the original label
                track = torch.stack([original_track, shifted_track], dim=0)
                return track, label
        else:
            # No transform, return original
            return original_track, label
