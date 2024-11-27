from pathlib import Path
from typing import Optional, Tuple, Union

import mirdata
import numpy as np
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset

from config.features import feature_configs


def pitch_to_midi(pitch_str: str) -> int:
    """Convert pitch string to MIDI note number."""
    # Define pitch classes
    pitch_classes = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }

    # Handle special cases for sharp/flat notes
    if len(pitch_str) == 3:
        pitch_class = pitch_str[:2]  # e.g., 'A#' from 'A#3'
        octave = int(pitch_str[2])
    else:
        pitch_class = pitch_str[0]  # e.g., 'A' from 'A3'
        octave = int(pitch_str[1])

    # Calculate MIDI note number
    # MIDI note numbers start at C-1 (octave = -1) which is MIDI note 0
    midi_note = pitch_classes[pitch_class] + (octave + 1) * 12

    return midi_note


class TinySOL(Dataset):
    def __init__(
        self,
        feature: str,
        root: Union[str, Path] = "data/TinySOL",
        split: Optional[str] = None,
        download: bool = False,
        feature_config: Optional[dict] = None,
        audio_format: str = "wav",
        item_format: str = "feature",
        fv: str = "pitch",
        seed: int = 42,
    ) -> None:
        """
        TinySOL dataset implementation.

        Args:
            feature: If split is None, prepare dataset for this feature extractor.
                        If split is not None, load these extracted features.
            root: Root directory of the dataset. Defaults to "data/MagnaTagATune".
            split: Split of the dataset to use: ["train", "test", "validation", None],
                        where None uses the full dataset (e.g. for feature extraction).
            download: Whether to download the dataset if it doesn't exist.
            feature_config: Configuration for the feature extractor.
            audio_format: Format of the audio files: ["mp3", "wav", "ogg"].
            item_format: Format of the items to return: ["audio", "feature"].
            fv: factor of variations (i.e. label) to return
            seed: Random seed for reproducibility.
        """
        self.name = "TinySOL"
        self.tasks = ["pitch_classification", "instrument_classification"]
        self.fvs = ["pitch", "instrument"]  # also dynamics, technique
        assert fv in self.fvs, f"Invalid factor of variation: {fv}"
        self.fv = fv

        root = Path(root)
        self.root = root
        if split not in [None, "train", "test", "validation"]:
            raise ValueError(
                f"Invalid split: {split} "
                + "Options: None, 'train', 'test', 'validation'"
            )
        self.split = split
        self.item_format = item_format
        self.audio_format = audio_format
        self.feature = feature

        if not feature_config:
            # load default feature config
            feature_config = feature_configs[feature]
        self.feature_config = feature_config

        # initialize mirdata dataset
        self.dataset = mirdata.initialize(dataset_name="tinysol", data_home=self.root)

        if download:
            self._download()

        self.label_encoder = LabelEncoder()

        self.paths, self.labels = self._load_metadata()

    def _download(self) -> None:
        self.dataset.download()
        self.dataset.validate(verbose=False)

    def _get_stratified_split(self, track_ids, labels, sizes=(0.8, 0.1, 0.1), seed=42):
        """Helper method to generate a stratified split of the dataset.

        Args:
            sizes (tuple, optional): Sizes of train, validation and test set.
                                     Defaults to (0.8, 0.1, 0.1), must add up to 1.
            seed (int, optional): Random seed. Defaults to 42.
        """
        if sum(sizes) != 1:
            raise ValueError("Sizes must add up to 1.")

        X_train, X_others, y_train, y_others = train_test_split(
            track_ids,
            labels,
            test_size=1 - sizes[0],
            random_state=seed,
            stratify=labels,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_others,
            y_others,
            test_size=sizes[2] / (sizes[1] + sizes[2]),
            random_state=seed,
            stratify=y_others,
        )
        return {
            "X_train": X_train,
            "X_validation": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_validation": y_val,
            "y_test": y_test,
        }

    def _load_metadata(self) -> Tuple[list, torch.Tensor]:
        # load track ids
        self.track_ids = self.dataset.track_ids
        # only keep track_ids with single pitch annotations
        for track_id in self.track_ids:
            if len(self.dataset.track(track_id).pitch) != 1:
                self.track_ids.remove(track_id)

        # load audio paths
        paths = [self.dataset.track(t_id).audio_path for t_id in self.track_ids]

        # load labels
        labels = []
        if self.fv == "pitch":
            for t_id in self.track_ids:
                pitch_str = self.dataset.track(t_id).pitch
                # annotation fix
                if "F#_" in pitch_str:
                    pitch_str = pitch_str.replace("F#_", "F#")
                labels.append(pitch_to_midi(pitch_str))
        elif self.fv == "instrument":
            for t_id in self.track_ids:
                labels.append(self.dataset.track(t_id).instrument_full)

        # load splits
        if self.split:
            splits = self._get_stratified_split(
                seed=42,
                track_ids=self.track_ids,
                labels=labels
            )
            paths, labels = splits[f"X_{self.split}"], splits[f"y_{self.split}"]

        # encode labels
        labels = self.label_encoder.fit_transform(labels)
        labels = torch.tensor(labels, dtype=torch.long)

        return paths, labels

    def load_track(self, path) -> Tensor:
        if self.item_format == "feature":
            return torch.load(path, weights_only=True)
        else:
            waveform, original_sample_rate = torchaudio.load(path, normalize=True)
            if waveform.size(0) != 1:  # make mono if stereo (or more)
                waveform = waveform.mean(dim=0, keepdim=True)
            if original_sample_rate != self.feature_config["sample_rate"]:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sample_rate,
                    new_freq=self.feature_config["sample_rate"],
                )
                waveform = resampler(waveform)
            return waveform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        path = self.paths[idx]
        labels = self.labels[idx]

        track = self.load_track(path)

        return track, labels
