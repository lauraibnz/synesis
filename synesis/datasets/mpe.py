import warnings
from pathlib import Path
from typing import Optional, Tuple, Union
import csv
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
import torch.nn.functional as F
import random
import copy
import collections
import os
import math

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
            root = kwargs["root"]
        
        self.root = Path(root)
        if split not in [None, "train", "validation", "test"]:
            raise ValueError(
                f"Invalid split: {split} "
                + "Options: 'None', 'train', 'validation', 'test'"
            )
        self.split = split
        self.item_format = item_format
        self.itemization = itemization
        self.feature = feature
        self.label = label
        self.feature_paths = None
        self.ext_audio = None
        self.ext_midi = None
        self.datasets = kwargs.get("datasets", None) # It will be None for training but paths will load the extracted data
        self.val_datasets = kwargs.get("val_datasets", None)
        self.test_datasets = kwargs.get("test_datasets", None)
        self.chunk_len_sec = kwargs.get("chunk_len_sec", 1.0)
        self.set_duration()
        self.hop = kwargs.get("hop", 0.8)
        # self.test_dataset = kwargs.get("test_dataset", "guitarset")
        # self.test_dataset_path = kwargs.get("test_dataset_path", None)
        self.feature_extractor = None
        self.hop_length = kwargs.get("hop_length", None)
        self.save_dir = "data/MPE"
        self.download = download # Always False for MPE dataset
        self.label_encoder = None # For compatibility with other datasets
        self.raw_data_paths = []
        self.sample_rate = kwargs.get("sample_rate", 44100)

        # create save directory if it doesn't exist
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # set the audio and midi file extensions
        self.set_extensions()

        seed_everything(0)

        self.paths = self.load_paths(self.load_split(self.save_dir, self.split)) if split else []
        self.feature_paths = self.paths

    def set_duration(self):
        if self.split == "train":
            self.duration_sec = 21600*6
        elif self.split == "validation":
            self.duration_sec = 2160*2
        elif self.split == "test":
            self.duration_sec = float('inf')
            #self.duration_sec = 2160

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        feature = np.load(self.paths[index])["feature"]
        label = np.load(self.paths[index])["label_frames"]
        if self.split != "test":
            mask_roll = np.load(self.paths[index])["mask_roll"]

        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        if self.split != "test":
            mask_roll = torch.tensor(mask_roll, dtype=torch.float32)

        # Note: Return mask later on when implementing custom BCE loss
        if self.split == "test":
            return (feature, label)
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
            
            if self.split is None:
                for each in ["train", "validation", "test"]:
                    self.split = each
                    self.set_duration()
                    self.extract_features()
            else:
                self.extract_features()
            self.paths = self.load_paths(self.load_split(self.save_dir, self.split)) if self.split else []
            self.feature_paths = self.paths
    
    def extract_features(self):
        """Extract features from raw audio files."""
        # Before extracting, create train, validation, test directories
        for split in ["train", "validation", "test"]:
            (Path(self.save_dir) / split).mkdir(parents=True, exist_ok=True)

        # Implement feature extraction logic here
        # Extract features from audio files
        total_samples_extracted = 0
        
        for each in self.get_datasets():
            audio_file = each[0]
            midi_file = each[1]
            assert audio_file.exists(), f"{audio_file} does not exist"
            assert midi_file.exists(), f"{midi_file} does not exist"

            # Load the MIDI file
            midi = pretty_midi.PrettyMIDI(str(midi_file))
            #midi = self.pedal_extend(str(midi_file))

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
            
            audio_len = waveform.shape[-1]
            
            if self.split !="test":
                chunk_len = int(self.chunk_len_sec * self.sample_rate)
                hop_samples = int(self.hop * self.sample_rate)
            else:
                chunk_len = int(self.chunk_len_sec * self.sample_rate)
                hop_samples = chunk_len//2
                self.hop = (hop_samples)/self.sample_rate
                #hop_samples = int((self.hop + 0.2) * self.sample_rate)
                # For test split, use the entire audio file
                #chunk_len = waveform.shape[-1]
                #hop_samples = chunk_len  # No overlap for test
                #self.chunk_len_sec = chunk_len / self.sample_rate

            # Fix padding logic
            remainder = (waveform.shape[-1] - chunk_len) % hop_samples
            if remainder != 0:
                pad_amount = hop_samples - remainder
                waveform = F.pad(waveform, (0, pad_amount), "constant", 0)

            num_chunks = int((waveform.shape[-1] - chunk_len) // hop_samples + 1)

            if self.split == "test":
                feat_chunks = []
                label_chunks = []
                audio_chunks = []

            for i in tqdm(range(num_chunks), desc=f"Extracting features... ", total=num_chunks):
                if (total_samples_extracted / self.sample_rate) > self.duration_sec:
                    print(f"Total hours extracted {total_samples_extracted/self.sample_rate/3600} exceeds duration {\
                        self.duration_sec/3600}.")
                    return
                
                start = i * self.hop
                end = start + self.chunk_len_sec

                # I use the approach below to round stuff and avoid floating point issues
                start_samples = int((start * self.sample_rate)+0.5)
                end_samples = int((end * self.sample_rate)+0.5)

                assert end_samples - start_samples == chunk_len, \
                    f"Chunk length mismatch: {end_samples - start_samples} != {chunk_len}"

                waveform_chunk = waveform[:, start_samples:end_samples]
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
                    #frame_rate = (feature.shape[1] - 1) / self.chunk_len_sec
                    frame_rate = feature.shape[1] / self.chunk_len_sec

                label_dict = self.get_label_roll(midi, self.chunk_len_sec, \
                    self.hop, i, frame_rate)
                

                # Ensure number of time steps of feature matches label_frames
                if label_dict["label_frames"].shape[0] > feature.shape[1]:
                    label_dict["label_frames"] = label_dict["label_frames"][:feature.shape[1], :]
                    label_dict["mask_roll"] = label_dict["mask_roll"][:feature.shape[1], :]
                    assert label_dict["label_frames"].shape[0] == feature.shape[1], \
                        f"Something is wrong! label_frames should match feature time steps {feature.shape[1]}."
                else:
                    assert label_dict["label_frames"].shape[0] <= feature.shape[1], \
                        f"Label frames {label_dict['label_frames'].shape[0]} exceeds feature time steps {feature.shape[1]}."

                if self.split != "test":
                    if "slakh" not in str(audio_file):
                        store_path = f"./{self.save_dir}/{self.split}/{str(audio_file.stem)}_{i}.npz"
                    else:
                        track_name = audio_file.parent.stem
                        store_path = f"./{self.save_dir}/{self.split}/{track_name}_{str(audio_file.stem)}_{i}.npz"
                    store_dict['audio'] = waveform_chunk
                    store_dict['feature'] = feature

                    # Store the label information as well
                    for key in label_dict.keys():
                        store_dict[key] = label_dict[key]
                        
                    np.savez(store_path, **store_dict)
                else:
                    feat_chunks.append(feature)
                    label_chunks.append(label_dict["label_frames"])
                    audio_chunks.append(waveform_chunk)
            
            if self.split == "test":
                feature = np.array(feat_chunks).squeeze()
                label_frames = np.array(label_chunks).squeeze()
                audio_chunks = np.array(audio_chunks).squeeze()

                if "slakh" not in str(audio_file):
                        store_path = f"./{self.save_dir}/{self.split}/{str(audio_file.stem)}.npz"
                else:
                    track_name = audio_file.parent.stem
                    store_path = f"./{self.save_dir}/{self.split}/{track_name}_{str(audio_file.stem)}.npz"
                store_dict['audio'] = audio_chunks
                store_dict['feature'] = feature

                # Store the label information as well
                store_dict['label_frames'] = label_frames
                store_dict['audio_len'] = audio_len
                    
                np.savez(store_path, **store_dict)
     
        print(f"Total samples extracted: {total_samples_extracted}")
        print(f"Features saved to {self.save_dir} directory.")
    
    def pedal_extend(self, midi_path: str, CC_SUSTAIN=64):
        """
            Extend the sustain events in the MIDI file.
            
            Args:
                midi_path (str): Path to the MIDI file.
                CC_SUSTAIN (int): MIDI control change number for sustain pedal. Default is 64.
            
            Returns:
                midi_copy (pretty_midi.PrettyMIDI): MIDI file with extended sustain events.
        """
        # make a copy of the MIDI file
        midi = pretty_midi.PrettyMIDI(midi_path)
        midi_copy = copy.deepcopy(midi)

        
        # we want to deal with events in the following order:
        # PEDAL_DOWN, PEDAL_UP, ONSET, OFFSET
        PEDAL_DOWN = 0
        PEDAL_UP = 1
        ONSET = 2
        OFFSET = 3

        # we will store all pedal and note events in a list
        # it will be sorted by time (pedal down, pedal up, onset, offset)
        events = []
        events.extend([(note.start, ONSET, [note, instrument]) for \
                        instrument in midi_copy.instruments for note in instrument.notes])
        events.extend([(note.end, OFFSET, [note, instrument]) for \
                        instrument in midi_copy.instruments for note in instrument.notes])
        
        for instrument in midi_copy.instruments:
            if not instrument.is_drum:
                for cc in instrument.control_changes:
                    if cc.number == CC_SUSTAIN:
                        if cc.value >= 64:
                            events.append((cc.time, PEDAL_DOWN, [cc, instrument]))
                        else:
                            events.append((cc.time, PEDAL_UP, [cc, instrument]))

        # sort the events by time and event type
        events.sort(key=lambda x: (x[0], x[1]))

        # We will keep a track of notes to extend (notes that fall within (pedal_down, pedal_up))
        # for each instrument
        extend_insts = collections.defaultdict(list)
        sus_insts = collections.defaultdict(bool) # stores sustain state for each instrument

        # We go through the events and extend the notes
        time = 0
        for time, event_type, event in events:
            if event_type == PEDAL_DOWN:
                sus_insts[event[1]] = True
            elif event_type == PEDAL_UP:
                sus_insts[event[1]] = False

                # If the pedal is up, we will end all of the notes
                # currently being extended
                ext_notes_inst = [] # Store new notes to extend
                for note in extend_insts[event[1]]:
                    if note.end < time:
                        # This note was extended, so we can end it
                        note.end = time
                    else:
                        # This note has not ended, so we still keep it
                        ext_notes_inst.append(note)
                extend_insts[event[1]] = ext_notes_inst
            elif event_type == ONSET:
                if sus_insts[event[1]]:
                    ext_notes_inst = []
                    # if sustain is on, we have to stop notes that are currently being extended
                    for note in extend_insts[event[1]]:
                        if note.pitch == event[0].pitch:
                            note.end = time
                            if note.start == note.end:
                                # it means this note now has zero duration,
                                # according to the official implementation, we should not keep it
                                event[1].notes.remove(note)
                        else:
                            # if the note is not the same as the one being extended,
                            # we can just add it to the list of notes to extend
                            ext_notes_inst.append(note)
                
                # Add the new set of notes to extend to the list
                extend_insts[event[1]].append(event[0])
            elif event_type == OFFSET:
                # if susain is on, we will not end the note
                # so let's consider where sustain is off
                if not sus_insts[event[1]]:
                    if event[0] in extend_insts[event[1]]:
                        extend_insts[event[1]].remove(event[0])
            else:
                raise AssertionError(f"Unknown event type: {event_type}")    
        
        # End notes that are still being extended
        for instrument in extend_insts.values():
            for note in instrument:
                note.end = time
        
        return midi_copy
    
    def load_split(self, save_dir, split):
        """
            Loads file paths from splits.txt for the requested split.
            
            Args:
                save_dir (str or Path): Directory where 'splits.txt' is located.
                split (str): One of 'train', 'validation', or 'test'.
                
            Returns:
                List of pathlib.Path: List of file paths for the requested split.
        """
        splits_path = Path(save_dir) / f"{split}"
        result = sorted(splits_path.rglob("*.npz"))
        if not result:
            raise FileNotFoundError(f"No files found for split: {split}")

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
                  duration: int | float, hop_size: float, idx: int, pr_rate, pitch_offset=0) -> dict:
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
        label_frames = np.zeros((num_frames, 128))
        mask_roll = np.ones((num_frames, 128))

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
            #assert audio_files[idx].stem == midi_files[idx].stem, \
            #    f"Audio file name: {audio_files[idx].stem} not the same as midi file: {midi_files[idx].stem}"

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

        if self.split == "train":
            split = "train"
        elif self.split == "validation":
            split = "validation"
        elif self.split == "test":
            split = "test"

        base_path = Path(path)/f"{split}/"
        tracks = [folder for folder in base_path.iterdir() if folder.is_dir()]
        for track in tqdm(tracks):
            try:
                audio_file = track / "mix.flac"
                midi_file = track / "all_src.mid"
                assert audio_file.exists(), f"{audio_file} does not exist"
                assert midi_file.exists(), f"{midi_file} does not exist"
                audio_files.append(audio_file)
                midi_files.append(midi_file)
                # metadata = track / "metadata.yaml"
                # with open(metadata, "r") as f:
                #     yaml_data = yaml.safe_load(f)
    
                # for key, value in yaml_data["stems"].items():
                #     if value["inst_class"] not in unwanted:
                #         audio_file = track / "stems" / f"{key}.flac"
                #         midi_file = track / "MIDI" / f"{key}.mid"

                #         try:
                #             assert audio_file.exists(), f"{audio_file} does not exist"
                #             assert midi_file.exists(), f"{midi_file} does not exist"
                #         except AssertionError as e:
                #             continue
                #         audio_files.append(audio_file)
                #         midi_files.append(midi_file)
            except:
                print(f"Error in {track}")
                continue
        
        # Check if the audio and midi files are the same
        # for the Slakh dataset
        self.checker_guitarset_slakh(audio_files, midi_files, dataset="slakh")
        files = list(zip(audio_files, midi_files))
        return files
    
    
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
    
    def get_maestro_train_val_test(self, base_path, ext_audio="wav", ext_midi="midi"):
        train_files = []
        val_files = []
        test_files = []
        split = self.split

        # metadata_csv is structured as follows:
        # canonical_composer, canonical_title, split, year, midi_filename, audio_filename, duration
        # read the csv file
        metadata_csv = Path(base_path) / "maestro-v3.0.0.csv"
        assert metadata_csv.exists(), f"{metadata_csv} does not exist"
        
        with open(metadata_csv, 'r') as f:
            content = csv.reader(f, delimiter=',', quotechar='"')

            base_path = Path(base_path)
            next(content)  # skip the header

            for i, each in enumerate(content):
                if each[2] == 'train':
                    midi_path = base_path / each[4]
                    audio_path = str(base_path / each[4]).replace(f".{ext_midi}", f".{ext_audio}")
                    audio_path = Path(audio_path)
                    train_files.append((audio_path, midi_path))
                    assert audio_path.exists(), f"{audio_path} does not exist"
                    assert midi_path.exists(), f"{midi_path} does not exist"
                elif each[2] == 'validation':
                    midi_path = base_path / each[4]
                    audio_path = str(base_path / each[4]).replace(f".{ext_midi}", f".{ext_audio}")
                    audio_path = Path(audio_path)
                    val_files.append((audio_path, midi_path))
                    assert audio_path.exists(), f"{audio_path} does not exist"
                    assert midi_path.exists(), f"{midi_path} does not exist"
                elif each[2] == 'test':
                    midi_path = base_path / each[4]
                    audio_path = str(base_path / each[4]).replace(f".{ext_midi}", f".{ext_audio}")
                    audio_path = Path(audio_path)
                    test_files.append((audio_path, midi_path))
                    assert audio_path.exists(), f"{audio_path} does not exist"
                    assert midi_path.exists(), f"{midi_path} does not exist"
                else:
                    raise ValueError(f"Split {each[2]} not supported")
        
        # Ensure train files is not in val_files or test_files, else throw an error
        # for train in train_files:
        #     if train in val_files or train in test_files:
        #         raise ValueError(f"Duplicate found: {train}")
        
        if split == "train":
            return train_files
        elif split == "validation":
            return val_files
        elif split == "test":
            return test_files
        else:
            raise ValueError(f"Split {split} not supported")
        # else:
        #     chosen_test_files = [
        #         "2009/MIDI-Unprocessed_02_R1_2009_03-06_ORIG_MID--AUDIO_02_R1_2009_02_R1_2009_04_WAV.midi",
        #         "2004/MIDI-Unprocessed_SMF_17_R1_2004_03-06_ORIG_MID--AUDIO_20_R2_2004_12_Track12_wav--1.midi",
        #         "2008/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--2.midi",
        #         "2014/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.midi",
        #         "2014/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--1.midi",
        #         "2008/MIDI-Unprocessed_10_R2_2008_01-05_ORIG_MID--AUDIO_10_R2_2008_wav--2.midi",
        #         "2018/MIDI-Unprocessed_Recital17-19_MID--AUDIO_18_R1_2018_wav--3.midi",
        #         "2008/MIDI-Unprocessed_10_R2_2008_01-05_ORIG_MID--AUDIO_10_R2_2008_wav--1.midi",
        #         "2008/MIDI-Unprocessed_14_R1_2008_01-05_ORIG_MID--AUDIO_14_R1_2008_wav--3.midi",
        #         "2013/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_08_R1_2013_wav--5.midi",
        #         "2015/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_03_R2_2015_wav--1.midi",
        #         "2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--6.midi"
        #     ]
        #     final_test_files = []

        #     for each in chosen_test_files:
        #         for file in test_files:
        #             if each in str(file[1]):
        #                 final_test_files.append((file[0], file[1]))
        #     return final_test_files

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
                midi.write(str(save_path) + f".mid")

        # Get the list of audio and midi
        audio_files = sorted((Path(path)/"audio_mono-mic").rglob(f"*.wav"))
        midi_files = sorted((Path(path)/"annotations-midi").rglob(f"*.mid"))

        # Use the GuitarSet checker to check if the audio and midi files are okay
        self.checker_guitarset_slakh(audio_files, midi_files)

        train_files = []
        val_files = []
        test_files = []

        train_len = int(0.8 * len(audio_files))
        val_len = int(0.1 * len(audio_files))

        train_files = list(zip(audio_files[:train_len], midi_files[:train_len]))
        val_files = list(zip(audio_files[train_len:train_len+val_len], midi_files[train_len:train_len+val_len]))
        test_files = list(zip(audio_files[train_len+val_len:], midi_files[train_len+val_len:]))

        # Ensure train files is not in val_files or test_files, else throw an error
        for train in train_files:
            if train in val_files or train in test_files:
                raise ValueError(f"Duplicate found: {train}")
        
        # Check val_files are not in test_files
        for val in val_files:
            if val in test_files:
                raise ValueError(f"Duplicate found: {val}")
        
        if self.split == "train":
            return train_files
        elif self.split == "validation":
            return val_files
        elif self.split == "test":
            return test_files

    def get_files_musicnet(self, path: str) -> tuple[list[Path], list[Path]]:
        train_files = []
        test_files = []
        val_files = []

        train_audio = sorted((Path(path)/"train_data/").rglob(f"*.wav"))
        train_midi = sorted((Path(path)/"train_labels/").rglob(f"*.mid"))
        test_audio = sorted((Path(path)/"test_data/").rglob(f"*.wav"))
        test_midi = sorted((Path(path)/"test_labels/").rglob(f"*.mid"))

        # The MIDI files have the right name but the wrong parent so we change it
        for i, midi_file in enumerate(train_midi):
            stem = midi_file.stem
            train_midi[i] = Path(path)/"musicnet_em"/f"{stem}.mid"

        for i, midi_file in enumerate(test_midi):
            stem = midi_file.stem
            test_midi[i] = Path(path)/"musicnet_em"/f"{stem}.mid"

        temp = []  
        temp.extend(list(zip(train_audio, train_midi)))

        # store 10% of train files as val files
        val_len = int(0.1 * len(temp))
        val_files.extend(temp[:val_len])
        train_files.extend(temp[val_len:])
        test_files.extend(list(zip(test_audio, test_midi)))

        # check paths are valid
        for audio, midi in train_files:
            assert audio.stem == midi.stem, f"Audio and MIDI files do not match: {audio}, {midi}"
        for audio, midi in val_files:
            assert audio.stem == midi.stem, f"Audio and MIDI files do not match: {audio}, {midi}"
        for audio, midi in test_files:
            assert audio.stem == midi.stem, f"Audio and MIDI files do not match: {audio}, {midi}"

        if self.split == "train":
            return train_files
        elif self.split == "validation":
            return val_files
        elif self.split == "test":
            return test_files

    def get_datasets(self):
        files = []
        if self.split == "train":
            dataset_items = self.datasets.items()
        elif self.split == "validation":
            dataset_items = self.val_datasets.items()
        elif self.split == "test":
            dataset_items = self.test_datasets.items()
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be one of 'train', 'validation', or 'test'.")


        for (dataset_type, path) in dataset_items:
            if dataset_type == "maestro":
                files.extend(self.get_maestro_train_val_test(path))
            elif dataset_type == "slakh":
                files.extend(self.get_files_slakh(path))
            elif dataset_type == "musicnet":
                files.extend(self.get_files_musicnet(path))
            elif dataset_type == "guitarset":
                files.extend(self.get_files_guitarset(path))
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}. \
                    Supported types: 'slakh', 'musicnet', 'guitarset'.")

        # if self.split != "test":
        #     for dataset_type, path in self.datasets.items():
        #             if dataset_type == "maestro":
        #                 files.extend(self.get_maestro_train_val_test(path))
        #             elif dataset_type == "slakh":
        #                 files.extend(self.get_files_slakh(path))
        #             elif dataset_type == "musicnet":
        #                 files.extend(self.get_files_musicnet(path))
        #             elif dataset_type == "guitarset":
        #                 files.extend(self.get_files_guitarset(path))
        #             else:
        #                 raise ValueError(f"Unsupported dataset type: {dataset_type}. \
        #                     Supported types: 'slakh', 'musicnet', 'guitarset'.")
        # else:
        #     #path = self.datasets[self.test_dataset]
        #     path = self.test_dataset_path
        #     if self.test_dataset == "maestro":
        #         files.extend(self.get_maestro_train_val_test(path))
        #     elif self.test_dataset == "slakh":
        #         files.extend(self.get_files_slakh(path))
        #     elif self.test_dataset == "musicnet":
        #         files.extend(self.get_files_musicnet(path))
        #     elif self.test_dataset == "guitarset":
        #         files.extend(self.get_files_guitarset(path))
        #     else:
        #         raise ValueError(f"Unsupported test dataset type: {self.test_dataset}. \
        #             Supported types: 'slakh', 'musicnet', 'guitarset'.")
        
        # randomly shuffle the files
        #random.seed(0)
        random.shuffle(files)
        return files