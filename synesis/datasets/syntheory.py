import csv
import os
import os.path
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union
import json
import sys

import numpy as np
import pandas as pd
import torch
import wget
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from config.features import configs as feature_configs
from synesis.datasets.dataset_utils import load_track

from ss_vq_vae.models import triplet_network


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
        timbre_model_path: Optional[str] = "./externals/ss-vq-vae/experiments/timbre_metric/checkpoint.ckpt",  # Path to timbre similarity model
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
            timbre_model_path: Path to the timbre similarity model checkpoint.
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
        self.timbre_model_path = timbre_model_path
        self.timbre_model = None  # Will be loaded lazily
        self.dataset_dir = None
        
        # Constants for timbre similarity
        self.SR = 16000
        self.MFCC_KWARGS = dict(n_mfcc=13, hop_length=500)
        
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

    def _load_timbre_model(self):
        """Load the timbre similarity model lazily."""
        if self.timbre_model is None and self.timbre_model_path is not None:
            try:
                timbre_model, triplet_backbone = triplet_network.build_model(num_features=12)
                timbre_model.load_weights(self.timbre_model_path)
                self.timbre_model = timbre_model
            except ImportError as e:
                raise ImportError(f"Could not import ss_vq_vae. Make sure it's available at {ss_vq_vae_path}. Error: {e}")
            except Exception as e:
                raise RuntimeError(f"Could not load timbre model from {self.timbre_model_path}. Error: {e}")

    def _extract_mfcc(self, audio):
        """Extract MFCC features, skipping the first coefficient (energy)"""
        return librosa.feature.mfcc(y=audio, sr=self.SR, **self.MFCC_KWARGS)[1:]

    def _pad_or_truncate(self, audio, ref):
        """Align audio lengths by padding or truncating"""
        if len(audio) < len(ref):
            return np.pad(audio, (0, len(ref) - len(audio)))
        return audio[:len(ref)]

    def _compute_timbre_similarity(self, audio_path1, audio_path2):
        """
        Compute timbre similarity between two audio files using trained model
        
        Args:
            audio_path1: Path to first audio file
            audio_path2: Path to second audio file
            
        Returns:
            float: Similarity score (0-1, higher = more similar)
        """
        if self.timbre_model_path is None:
            raise ValueError("timbre_model_path must be provided to compute timbre similarity")
        
        # Load model if not already loaded
        self._load_timbre_model()
        
        # Load audio files
        audio1, _ = librosa.load(audio_path1, sr=self.SR)
        audio2, _ = librosa.load(audio_path2, sr=self.SR)
        
        # Align lengths
        audio2 = self._pad_or_truncate(audio2, audio1)
        audio1 = self._pad_or_truncate(audio1, audio2)
        
        # Extract MFCC features
        mfcc1 = self._extract_mfcc(audio1)
        mfcc2 = self._extract_mfcc(audio2)
        
        # Prepare triplet format and compute similarity
        sim = self.timbre_model.predict([
            mfcc1.T[None, :, :],  # Reference
            mfcc2.T[None, :, :],  # Query
            mfcc2.T[None, :, :]   # Duplicate query for triplet format
        ], verbose=0)[0][0]  # Extract cosine similarity score
        
        return sim

    def _load_metadata(self) -> None:
        """Load metadata and create train/test/validation splits."""
        # Load the info.csv file for this specific concept
        # Special handling for instruments which uses the notes dataset
        if self.label == "instruments" or self.label == "categories":
            self.dataset_dir = "chords"
        else:
            self.dataset_dir = self.label
            
        info_path = self.root / self.dataset_dir / "info.csv"
        if not info_path.exists():
            raise FileNotFoundError(f"Info file not found: {info_path}")
        
        # Load the dataset info
        df = pd.read_csv(info_path)
        
        # Store the full dataframe for finding similar samples when transforms are applied
        self.metadata_df = df
        
        # Ensure we have the required columns
        required_cols = ["synth_file_path", self.target_column]
        if self.dataset_dir == "tempos":
            required_cols[0] = "offset_file_path"
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in info.csv")
        
        # Create full audio paths
        audio_paths = []
        path_col = "offset_file_path" if self.dataset_dir == "tempos" else "synth_file_path"
        for audio_path in df[path_col]:
            full_path = self.root / self.dataset_dir / audio_path
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
    
        # Ensure at least 2 samples with same content but different instruments per split
        # For notes, chords, and instruments (which all have multiple instruments per content)
        if self.dataset_dir in ["notes", "chords", "tempos"]:
            if self.dataset_dir == "notes":
                # Group by note + octave + register
                df['content_key'] = df['root_note_pitch_class'].astype(str) + '_' + df['octave'].astype(str) + '_' + df['register'].astype(str)
            elif self.dataset_dir == "chords":
                # Group by chord type + root note
                df['content_key'] = df['chord_type'].astype(str) + '_' + df['root_note_pitch_class'].astype(str)
            elif self.dataset_dir == "tempos":
                # Group by bpm and discrete offset index parsed from filename to avoid float mismatch
                if 'offset_file_path' not in df.columns:
                    raise ValueError("Required column for tempo grouping not found: 'offset_file_path'")
                # Extract offset_id from filenames like ..._offset_3.wav
                offset_ids = df['offset_file_path'].str.extract(r'_offset_(\d+)')[0]
                if offset_ids.isnull().any():
                    raise ValueError("Failed to parse offset_id from 'offset_file_path' for some rows")
                df['offset_id'] = offset_ids.astype(int)
                df['content_key'] = df['bpm'].astype(str) + '_' + df['offset_id'].astype(str)
            
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
                
                # Shuffle instruments for this content deterministically (seed set above)
                np.random.shuffle(group_indices)
                
                # Collapse duplicates of the same instrument id (e.g., multiple samples of same instrument)
                # This ensures each content group has exactly one sample per unique instrument for clean pairing
                seen_instr = set()
                primary_indices = []
                duplicate_indices = []
                for gi in group_indices:
                    instr_id = int(df.loc[gi, 'midi_program_num'])
                    if instr_id in seen_instr:
                        duplicate_indices.append(gi)  # Duplicates become leftovers
                    else:
                        seen_instr.add(instr_id)
                        primary_indices.append(gi)    # Only one per instrument ID
                
                # Form pairs from primary indices (groups of 2). Leftover (odd count) handled later
                num_pairs = len(primary_indices) // 2
                pairs = [primary_indices[i * 2:(i + 1) * 2] for i in range(num_pairs)]
                
                # For each content group (bpm/offset), decide which split gets the pair
                # This ensures each split can have valid InstrumentShift pairs for some content
                for pair in pairs:
                    # Randomly assign each pair to a split with 70/15/15 probability
                    split_choice = np.random.choice(['train', 'test', 'validation'], p=[0.7, 0.15, 0.15])
                    if split_choice == 'train':
                        train_indices.extend(pair)
                    elif split_choice == 'test':
                        test_indices.extend(pair)
                    else:  # validation
                        val_indices.extend(pair)

                # Handle leftovers: any odd primary and all duplicates
                remaining_indices = primary_indices[num_pairs * 2:]
                for remaining_idx in remaining_indices:
                    split_choice = np.random.choice(['train', 'test', 'validation'], p=[0.7, 0.15, 0.15])
                    if split_choice == 'train':
                        train_indices.append(remaining_idx)
                    elif split_choice == 'test':
                        test_indices.append(remaining_idx)
                    else:  # validation
                        val_indices.append(remaining_idx)
                
                # Handle duplicates: assign them to the same split as their corresponding primary
                for duplicate_idx in duplicate_indices:
                    duplicate_instr_id = int(df.loc[duplicate_idx, 'midi_program_num'])
                    # Find which split the primary with this instrument ID went to
                    primary_split = None
                    if duplicate_idx in train_indices:
                        primary_split = 'train'  # Already assigned somehow
                    else:
                        # Find the primary with the same instrument ID and see which split it's in
                        for pi in primary_indices:
                            if int(df.loc[pi, 'midi_program_num']) == duplicate_instr_id:
                                if pi in train_indices:
                                    primary_split = 'train'
                                elif pi in test_indices:
                                    primary_split = 'test'
                                elif pi in val_indices:
                                    primary_split = 'validation'
                                break
                    
                    # Assign duplicate to the same split as its primary
                    if primary_split == 'train':
                        train_indices.append(duplicate_idx)
                    elif primary_split == 'test':
                        test_indices.append(duplicate_idx)
                    elif primary_split == 'validation':
                        val_indices.append(duplicate_idx)
                    else:
                        # Fallback: random assignment if primary not found
                        split_choice = np.random.choice(['train', 'test', 'validation'], p=[0.7, 0.15, 0.15])
                        if split_choice == 'train':
                            train_indices.append(duplicate_idx)
                        elif split_choice == 'test':
                            test_indices.append(duplicate_idx)
                        else:  # validation
                            val_indices.append(duplicate_idx)

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
        if self.dataset_dir == "notes":
            mask = (
                (current_split_df['root_note_pitch_class'] == current_row['root_note_pitch_class']) &
                (current_split_df['octave'] == current_row['octave']) &
                (current_split_df['register'] == current_row['register']) &
                (current_split_df['midi_program_num'] != current_instrument)
            )
        # For chords dataset, we want same chord type, root note but different instrument
        elif self.dataset_dir == "chords":
            mask = (
                (current_split_df['root_note_pitch_class'] == current_row['root_note_pitch_class']) &
                (current_split_df['chord_type'] == current_row['chord_type']) &
                (current_split_df['midi_program_num'] != current_instrument)
            )
        # For tempos dataset, we want same bpm and offset_time but different instrument
        elif self.dataset_dir == "tempos":
            mask = (
                (current_split_df['bpm'] == current_row['bpm']) &
                (current_split_df['offset_id'] == current_row['offset_id']) &
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
                # For equivariance: return the timbre similarity as transform parameter
                if self.item_format == "raw":
                    # Compute timbre similarity between original and shifted audio files
                    original_audio_path = self.raw_data_paths[idx]
                    shifted_audio_path = self.raw_data_paths[shifted_idx]
                    timbre_similarity = self._compute_timbre_similarity(original_audio_path, shifted_audio_path)
                    
                    transform_param = torch.tensor(timbre_similarity, dtype=torch.float32)
                else:
                    # For feature format, we can't compute timbre similarity directly
                    # Fall back to one-hot encoding difference
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
