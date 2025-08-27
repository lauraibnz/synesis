import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
import yaml
import pretty_midi
import numpy as np
from pathlib import Path
import jams
from config.features import configs as feature_configs

class MPE(Dataset):
    def __init__(self,
        feature: str,
        root: Union[str, Path] = "/home/nkcemeka/Documents/Datasets/maestro-v3.0.0",
        split: Optional[str] = None,
        feature_config: Optional[dict] = None,
        item_format: str = "feature",
        itemization: bool = True,
        label: str = "mpe",
        download: bool = False,
        **kwargs) -> None:
        """
            MPE dataset implementation. 

            Args:
                feature (str): Feature extraction model name.
                root (str or Path): Root directory of the dataset. We assume the dataset is already downloaded.
                split (str, optional): Dataset split to use. Options: None, 'train', 'validation', 'test'.
                feature_config (dict, optional): Configuration for the feature extractor.
                item_format (str): Format of the items in the dataset. Default is "feature".
                itemization (bool): Whether to itemize the dataset. Default is True.
                label (str): Label for the dataset. Default is "mpe".
                chunk_len_sec (float): Length of each audio chunk in seconds. Default is 1.0.
                duration_sec (float): Total duration of the dataset in seconds. Default is 27000.0.
                hop_length (int): Hop length in samples for feature extraction. Default is 256.
        """
        super().__init__() 
        
        if root is None:
            raise ValueError("Root directory must be specified for MPE dataset.")

        if kwargs.get("root"):
            root = self.kwargs["root"]
        
        self.root = Path(root)
        if split not in [None, "train", "validation", "test"]:
            raise ValueError(
                f"Invalid split: {split} "
                + "Options: None, 'train', 'validation', 'test'"
            )
        self.split = split
        self.item_format = item_format
        self.itemization = itemization
        self.feature = feature
        self.label = label
        self.feature_paths = None
        self.ext_audio = None
        self.ext_midi = None
        self.chunk_len_sec = kwargs.get("chunk_len_sec", 1.0)
        self.duration_sec = kwargs.get("duration_sec", 27000.0)
        self.feature_extractor = None
        self.hop_length = kwargs.get("hop_length", None)
        self.save_dir = "data/MPE"
        self.download = download # Always False for MPE dataset
        self.label_encoder = None # For compatibility with other datasets
        self.raw_data_paths = []
        self.sample_rate = kwargs.get("sample_rate", 16000)
        self.dataset_type = kwargs.get("dataset_type", "maestro")

        # create save directory if it doesn't exist
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # set the audio and midi file extensions
        self.set_extensions()

        if not feature_config:
            feature_config = feature_configs[feature]
        self.feature_config = feature_config["extract_kws"]
        
        if self.dataset_type == "slakh":
            self.audio_files, self.midi_files = self.get_files_slakh(self.root)
        elif self.dataset_type == "maestro":
            self.audio_files, self.midi_files = self.get_files_maestro(self.root)
        elif self.dataset_type == "guitarset":
            self.audio_files, self.midi_files = self.get_files_guitarset(self.root)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}. Supported types: 'slakh', 'maestro', 'guitarset'.")
        
        print(f"Number of audio files: {len(self.audio_files)}, Number of midi files: {len(self.midi_files)}")

        self.paths = self.load_paths(self.load_split(self.save_dir, self.split)) if split else []
        self.feature_paths = self.paths
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        feature = np.load(self.paths[index])["feature"]
        label = np.load(self.paths[index])["label_frames"]
        mask_roll = np.load(self.paths[index])["mask_roll"]

        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        mask_roll = torch.tensor(mask_roll, dtype=torch.float32)

        # Note: Return mask later on when implementing custom BCE loss
        return (feature, [label, mask_roll])

    def set_extractor(self, extractor):
        """Set the feature extractor for the dataset."""
        self.feature_extractor = extractor
        self.prepare_dataset(extractor)
    
    def prepare_dataset(self, extractor=None):
        if not self.paths:
            print(f"No features found in {self.save_dir}. Extracting features...")
            if extractor is None:
                raise ValueError("Feature extractor must be provided for MPE dataset.")
            self.extract_features()
            self.paths = self.load_paths(self.load_split(self.save_dir, self.split)) if self.split else []
            self.feature_paths = self.paths
    
    def extract_features(self):
        """Extract features from raw audio files."""
        # Implement feature extraction logic here
        # Extract features from audio files
        total_samples_extracted = 0
        for audio_file, midi_file in zip(self.audio_files, self.midi_files):
            assert audio_file.exists(), f"{audio_file} does not exist"
            assert midi_file.exists(), f"{midi_file} does not exist"

            # Load the MIDI file
            midi = pretty_midi.PrettyMIDI(str(midi_file))

            # Load the audio file
            waveform, sample_rate = torchaudio.load(audio_file)
            
            if waveform.size(0) != 1:  # make mono if stereo (or more)
                waveform = waveform.mean(dim=0, keepdim=True)
        

            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.sample_rate,
                )
                waveform = resampler(waveform)
            chunk_len = int(self.chunk_len_sec * self.sample_rate)

            # Generate chunks of audio
            initial_num_chunks = waveform.size(1) // chunk_len
            remainder = waveform.size(1) % chunk_len
            chunks = []
            chunks.append(waveform[:, :initial_num_chunks * chunk_len])

            if remainder > 0:
                # pad the last chunk to length of chunk_len and append to chunks
                padded_chunk = torch.cat(
                    [waveform[:, initial_num_chunks * chunk_len:], 
                     torch.zeros(1, chunk_len - remainder)],
                    dim=1
                )
                chunks.append(padded_chunk)
            
            # concatenate chunks into one tensor
            waveform_chunks = torch.cat(chunks, dim=1)
            assert waveform_chunks.shape[1] % chunk_len == 0, \
                f"Waveform length {waveform_chunks.shape[1]} is not a multiple of chunk_len {chunk_len}"
            
            # Extract features
            for i in tqdm(range(int(waveform_chunks.shape[1] / chunk_len)), desc=f"Extracting features... ",\
                     total=int(waveform_chunks.shape[1] / chunk_len)):
                
                if (total_samples_extracted / self.sample_rate) > self.duration_sec:
                    print(f"Total hours extracted {total_samples_extracted/self.sample_rate/3600} exceeds duration {\
                        self.duration_sec/3600}.")
                    # Get splits
                    self.get_splits()
                    return
                
                start = i * chunk_len
                end = start + chunk_len
                waveform_chunk = waveform_chunks[:, start:end]
                feature = self.feature_extractor(waveform_chunk)
                feature = feature.squeeze().T.unsqueeze(0) # Transpose to (1, time, n_mels)
                total_samples_extracted += waveform_chunk.shape[1]

                # Get the label roll for the current chunk
                store_dict = {'audio': None, 'feature': None} 

                # since we know the hop length in samples (or dist btw two consecutive frames)
                # we can convert that to seconds using the sample rate
                if self.hop_length is not None:
                    frame_rate = self.sample_rate / self.hop_length
                else:
                    # This is not some spectrogram-like feature so we use the 
                    # feature shape to determine the frame rate
                    # why subtract 1? This is a hack to make things align: Still needs to be fixed
                    frame_rate = (feature.shape[1] - 1) / self.chunk_len_sec

                label_dict = self.get_label_roll(midi, self.chunk_len_sec, \
                    self.chunk_len_sec, i, frame_rate)
                

                # Ensure number of time steps of feature matches label_frames
                if label_dict["label_frames"].shape[0] != feature.shape[1]:
                    assert label_dict["label_frames"].shape[0] == feature.shape[1], \
                        f"Label frames {label_dict['label_frames'].shape[0]} does not match feature time steps {feature.shape[1]}."
                
                store_path = f"./{self.save_dir}/{str(audio_file.stem)}_{i}.npz" 
                store_dict['audio'] = waveform_chunk
                store_dict['feature'] = feature

                # Store the label information as well
                for key in label_dict.keys():
                    store_dict[key] = label_dict[key]
                    
                np.savez(store_path, **store_dict)
     
        print(f"Total samples extracted: {total_samples_extracted}")
        print(f"Features saved to {self.save_dir} directory.")
        self.get_splits()

    def get_splits(self):
        """
            Obtain the splits of the dataset and save them to a file.
            This is useful for datasets that do not have predefined splits.
        """
        # load all the file names of extracted features
        print(f"Obtaining splits...")
        temp_paths = sorted(Path(self.save_dir).glob(f"*.npz"))
        print(f"Number of extracted features: {len(temp_paths)}")

        # randomly shuffle the paths
        rng = np.random.default_rng(seed=42)
        rng.shuffle(temp_paths)

        # split according to 80/10/10 split
        train_split = int(len(temp_paths) * 0.8)
        train = temp_paths[:train_split]
        remainder = temp_paths[train_split:]
        val_split = int(len(remainder) * 0.5)
        val = remainder[:val_split]
        test = remainder[val_split:]

        # save the splits to a file
        with open(Path(self.save_dir) / "splits.txt", "w") as f:
            f.write("train:\n")
            for path in train:
                f.write(f"{path}\n")
            f.write("validation:\n")
            for path in val:
                f.write(f"{path}\n")
            f.write("test:\n")
            for path in test:
                f.write(f"{path}\n")
    
    def load_split(self, save_dir, split):
        """
            Loads file paths from splits.txt for the requested split.
            
            Args:
                save_dir (str or Path): Directory where 'splits.txt' is located.
                split (str): One of 'train', 'validation', or 'test'.
                
            Returns:
                List of pathlib.Path: List of file paths for the requested split.
        """
        splits_path = Path(save_dir) / "splits.txt"
        split = split.lower()
        assert split in {"train", "validation", "test"}, "Invalid split requested."

        result = []
        current_section = None
        try:
            with open(splits_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.endswith(":"):
                        current_section = line[:-1].lower()
                        continue
                    if current_section == split and line:
                        # Each line is a file path; Path(line) keeps it as Path
                        result.append(Path(line))
                    if line.endswith(":") and current_section != split:
                        continue
        except:
            raise FileNotFoundError(f"Splits file {splits_path} not found. Please run feature extraction first.")
        
        return result
    
    def load_paths(self, filenames):
        paths = []
        for filename in filenames:
            path = Path(filename)
            if path.exists():
                paths.append(path)
            else:
                warnings.warn(f"File {path} does not exist.")
        return paths
    
    def set_extensions(self):
        """
            Set the audio and midi file extensions.
        """
        audio_extensions = ["wav", "flac"]
        midi_extensions = ["midi", "mid"]
        
        for ext in audio_extensions:
            if sorted(Path(self.root).rglob(f"*.{ext}")):
                self.ext_audio = ext
                break
        
        for ext in midi_extensions:
            if sorted(Path(self.root).rglob(f"*.{ext}")):
                self.ext_midi = ext
                break
        
        print(f"Audio extension set to: {self.ext_audio}, MIDI extension set to: {self.ext_midi}")

    
    def get_label_roll(self, midi: pretty_midi.PrettyMIDI, \
                  duration: int | float, hop_size: float, idx: int, pr_rate, pitch_offset=21) -> dict:
        """
            Get the label and pedal rolls for a given audio segment.
            The regressed rolls generated follow Kong's model!

            Args:
                midi (pretty_midi.PrettyMIDI): PrettyMIDI object
                duration (float): Duration of the audio segment
                hop_size (float): Hop size in seconds
                idx (int): Index of the audio segment
                pr_rate (int): The frame rate of the piano roll

            Returns:
                target_dict: {
                    label_frames (np.ndarray): Frames label
                    mask_roll (np.ndarray): Mask roll label to remove all events that do not occur
                                            in the audio segment
                    note_events (np.ndarray): Array of note events for the audio segment
                } 
        """
        # init target_dict and set the keys to None
        target_dict: dict[str, Optional[np.ndarray]] = {
            'label_frames': None,
            'mask_roll': None,
            'note_events': None
        }

        num_frames = int(round(duration * pr_rate)) + 1
        start = idx * hop_size
        end = start + duration

        # initialize the labels
        label_frames = np.zeros((num_frames, 88))
        mask_roll = np.ones((num_frames, 88))

        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= end or note.end <= start:
                        continue 

                    pitch = note.pitch - pitch_offset
                    
                    start_frame = int(round(pr_rate * (note.start - start)))

                    # clamp end frame like was done in Kong's paper
                    end_frame = int(round(pr_rate * (note.end - start)))

                    # prepare labels (note that end_frame is >= 0 else it would
                    # have been skipped)
                    label_frames[max(0, start_frame):min(end_frame + 1, num_frames), pitch] = 1
                    if end_frame < num_frames:
                        if start_frame < 0:
                            # We will never get here
                            mask_roll[:end_frame + 1, pitch] = 0
                    else:
                        if start_frame >= 0:
                            # ------------------------------
                            mask_roll[start_frame:, pitch] = 0
                        else:
                            # we won't get here too
                            mask_roll[:, pitch] = 0

        # update the target_dict
        target_dict['label_frames'] = label_frames
        target_dict['mask_roll'] = mask_roll

        return target_dict
    
    def checker(self, audio_files: list[Path], midi_files: list[Path]) -> None:
        """
            Check if the audio and midi files are the same.
            This function should be used only on datasets
            with audio and midi files having the same name.

            Args:
                audio_files (list): List of audio files
                midi_files (list): List of midi files

            Returns:
                None
        """
        assert len(audio_files) == len(midi_files), \
              f"Number of audio files: {len(audio_files)} != Number of midi files: {len(midi_files)}"

    def checker_guitarset_slakh(self, audio_files: list[Path], midi_files: list[Path], \
                                dataset: str="guitarset") -> None:
        """
            Check if the audio and midi files are the same
            for the GuitarSet dataset (assuming the audio is
            from the audio_mono-mic folder) or for Slakh.

            Args:
                audio_files (list): List of audio files
                midi_files (list): List of midi files

            Returns:
                None
        """
        assert len(audio_files) == len(midi_files), \
              f"Number of audio files: {len(audio_files)} != Number of midi files: {len(midi_files)}"
        
        # Generate a random number from 0 to len(audio_files)
        idx = np.random.randint(0, len(audio_files))
        if dataset == "guitarset":
            assert audio_files[idx].stem[:-4] == midi_files[idx].stem, \
                f"Audio file name: {audio_files[idx].stem} not the same as midi file: {midi_files[idx].stem}"
        else:
            # We assume its Slakh
            audio_track_name = audio_files[idx].parent.parent.stem
            midi_track_name = midi_files[idx].parent.parent.stem

            # Check the track names
            assert audio_track_name == midi_track_name, \
                f"Audio Track name: {audio_track_name} not the same as midi Track name: {midi_track_name}"
            
            # Check the file names
            assert audio_files[idx].stem == midi_files[idx].stem, \
                f"Audio file name: {audio_files[idx].stem} not the same as midi file: {midi_files[idx].stem}"

    def get_files_slakh(self, path: str) -> tuple[list[Path], list[Path]]:
        """
        Get the list of audio and midi files from the given path
        for the Slakh dataset.

        Args:
            path (str): Path to the Slakh dataset

        Returns:
            audio_files (list): List of audio files
            midi_files (list): List of midi files
        """
        unwanted = ["Drums", "Percussive", "Sound Effects", "Sound effects", \
                    "Chromatic Percussion"]
        audio_files = []
        midi_files = []

        for each in ['train', 'validation', 'test']:
            base_path = Path(path)/f"{each}/"
            tracks = [folder for folder in base_path.iterdir() if folder.is_dir()]
            for track in tqdm(tracks):
                try:
                    metadata = track / "metadata.yaml"
                    with open(metadata, "r") as f:
                        yaml_data = yaml.safe_load(f)
        
                    for key, value in yaml_data["stems"].items():
                        if value["inst_class"] not in unwanted:
                            audio_file = track / "stems" / f"{key}.{self.ext_audio}"
                            midi_file = track / "MIDI" / f"{key}.{self.ext_midi}"

                            try:
                                assert audio_file.exists(), f"{audio_file} does not exist"
                                assert midi_file.exists(), f"{midi_file} does not exist"
                            except AssertionError as e:
                                continue
                            audio_files.append(audio_file)
                            midi_files.append(midi_file)
                except:
                    print(f"Error in {track}")
                    continue
        
        # Check if the audio and midi files are the same
        # for the Slakh dataset
        self.checker_guitarset_slakh(audio_files, midi_files, dataset="slakh")
        return audio_files, midi_files
    
    
    def get_files_maestro(self, path: str) -> tuple[list[Path], list[Path]]:
        """
            Get the list of audio and midi files from the given path
            for the MAESTRO dataset.

            Args:
                path (str): Path to the MAESTRO dataset

            Returns:
                audio_files (list): List of audio files
                midi_files (list): List of midi files
        """
        audio_files = sorted(Path(path).rglob(f"*.{self.ext_audio}"))
        midi_files = sorted(Path(path).rglob(f"*.{self.ext_midi}"))

        # Since MAESTRO's audio and midi files have the same name,
        # we can use checker
        self.checker(audio_files, midi_files)
        return audio_files, midi_files
    
    def jams_to_midi(self, jam: jams.JAMS, q: int = 1) -> pretty_midi.PrettyMIDI:
        """
            Convert jams to midi using pretty_midi.
            Gotten from the `marl repo`_.
            .. _marl repo: https://github.com/marl/GuitarSet/blob/master/visualize/interpreter.py

            Args:
                jam (jams.JAMS): Jams object
                q (int): 1: with pitch bend. q = 0: without pitch bend.
            
            Returns:
                midi: PrettyMIDI object
        """
        # q = 1: with pitch bend. q = 0: without pitch bend.
        midi = pretty_midi.PrettyMIDI()
        annos = jam.search(namespace='note_midi')
        if len(annos) == 0:
            annos = jam.search(namespace='pitch_midi')
        for anno in annos:
            midi_ch = pretty_midi.Instrument(program=25)
            for note in anno:
                pitch = int(round(note.value))
                bend_amount = int(round((note.value - pitch) * 4096))
                st = note.time
                dur = note.duration
                n = pretty_midi.Note(
                    velocity=100 + np.random.choice(range(-5, 5)),
                    pitch=pitch, start=st,
                    end=st + dur
                )
                pb = pretty_midi.PitchBend(pitch=bend_amount * q, time=st)
                midi_ch.notes.append(n)
                midi_ch.pitch_bends.append(pb)
            if len(midi_ch.notes) != 0:
                midi.instruments.append(midi_ch)
        return midi

    
    def get_files_guitarset(self, path: str) -> tuple[list[Path], list[Path]]:
        """
            Get the list of audio and midi files from the given path
            for the GuitarSet dataset.

            Args:
                path (str): Path to the GuitarSet dataset

            Returns:
                audio_files (list): List of audio files
                midi_files (list): List of midi files
        """
        # check if the annotations-midi folder exists
        if not (Path(path)/"annotations-midi").exists():
            # Get all the jams in this path
            path_annot = Path(path)/"annotation"
            all_jams = sorted(path_annot.glob("*.jams")) 

            for _, jamPath in tqdm(enumerate(all_jams), total=len(all_jams)):
                jam_path = str(jamPath)
                jam = jams.load(jam_path)
                midi = self.jams_to_midi(jam, q=1)
                save_path = path_annot.parent / f"annotations-midi/{Path(jam_path).stem}"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                midi.write(str(save_path) + f".{self.ext_midi}")

        # Get the list of audio and midi
        audio_files = sorted((Path(path)/"audio_mono-mic").rglob(f"*.{self.ext_audio}"))
        midi_files = sorted((Path(path)/"annotations-midi").rglob(f"*.{self.ext_midi}"))

        # Use the GuitarSet checker to check if the audio and midi files are okay
        self.checker_guitarset_slakh(audio_files, midi_files)

        return audio_files, midi_files