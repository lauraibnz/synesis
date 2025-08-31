import importlib
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler
from tqdm import tqdm

from config.features import configs as feature_configs


class FeatureExtractorFactory:
    @classmethod
    def get_feature_extractor(cls, name: str, **kwargs):
        """
        Retrieve a feature extractor instance based on its name and parameters.

        Args:
            name: The name of the feature extractor to retrieve.
            **kwargs: Optional parameters to pass to the constructor.

        Returns:
            An instance of the requested feature extractor.

        Raises:
            ValueError: If the feature extractor name is not recognized.
        """

        __cls__ = feature_configs[name]["__cls__"]
        extract_kws = feature_configs[name].get("extract_kws", None)
        kwargs.update({"extract_kws": extract_kws}) if extract_kws else None

        try:
            # Dynamically import the feature extractor module
            module = importlib.import_module(f"synesis.features.{__cls__.lower()}")
            # Get the feature extractor class
            extractor_class = getattr(module, __cls__)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Unknown feature extractor: {__cls__}") from e

        # Create instance with feature_extractor=True
        model = extractor_class(feature_extractor=True, **kwargs)

        # Load weights
        try:
            if os.path.exists(f"models/pretrained/{name.lower()}.pt"):
                weights_path = f"models/pretrained/{name.lower()}.pt"
            elif os.path.exists(f"models/pretrained/{name.lower()}.pth"):
                weights_path = f"models/pretrained/{name.lower()}.pth"
            else:
                weights_path = f"models/pretrained/{name.lower()}.ckpt"
            try:
                model.load_state_dict(torch.load(weights_path))
            except AttributeError as e:
                warnings.warn(f"Weights might not be loaded! Error: {e}", UserWarning)
        except FileNotFoundError:
            # print(f"No pretrained weights found for {name}.")
            # !TODO: streamline ckeckpoint loading
            pass
        try:
            model.eval()
        except AttributeError as e:
            warnings.warn(
                f"Model might not be set to eval mode! Error: {e}", UserWarning
            )

        return model


def get_feature_extractor(name: str, **kwargs):
    """
    Convenience function to get a feature extractor instance.

    Args:
        name: The name of the feature extractor to retrieve.
        **kwargs: Optional parameters to pass to the constructor.

    Returns:
        An instance of the requested feature extractor.
    """
    return FeatureExtractorFactory.get_feature_extractor(name, **kwargs)


def dynamic_batch_extractor(
    dataset,
    extractor,
    item_len: int,
    padding: str = "repeat",
    batch_size: int = 32,
    device: str = "cpu",
):
    """
    Create and process batches from a PyTorch dataset with items of
    variable length (such as audio tracks).

    Args:
        dataset: PyTorch dataset that returns [audio, label].
        item_len: Length of each eventual item (not item returned by dataset)
                  (e.g. in samples if audio).
        extractor: Function (or class with forward method) to process batches.
        padding: Padding method for items, either "repeat" or "zero".
        batch_size: Batch size.
    """

    def save_or_append(batch, batch_paths):
        for idx, output_path in enumerate(batch_paths):
            emb = batch[idx]
            if emb.shape[0] != 1:
                emb = emb.unsqueeze(0)

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(emb, path)

    # Identify if .pt files are already present, and if so, skip them.
    # We do this now as the dynamic extractor writes and loads .pt files,
    # concatenating new embeddings, so we don't want to interefere with that.
    existing_files = [p for p in dataset.feature_paths if Path(p).exists()]

    if hasattr(dataset, "set_extractor") and not dataset.feature_paths:
        # If the dataset has a set_extractor method, use it to set the extractor
        dataset.set_extractor(extractor)

    pbar = tqdm(total=len(dataset))
    batch = []
    batch_paths = []
    extractor.to(device)


    for i in range(len(dataset)):
        x, _ = dataset[i]  # Ignore the label
        output_path = dataset.feature_paths[i]
        # skip if output file exists
        if output_path in existing_files:
            pbar.update(1)
            continue

        for buffer in range(0, x.shape[1], item_len):
            x_item = x[:, buffer : buffer + item_len]
            if x_item.shape[1] < item_len:
                match padding:
                    case "repeat":
                        if x_item.shape[1] < item_len:
                            # repeat the first part of the item, and add it in front
                            # e.g. if x_item is [1, 2, 3] and item_len is 5, then
                            # the resulting x_item is [1, 2, 1, 2, 3]
                            repeated_part = x_item[:, : item_len - x_item.shape[1]]
                            while x_item.shape[1] < item_len:
                                x_item = torch.cat([repeated_part, x_item], dim=1)
                            x_item = x_item[:, :item_len]
                    case "zero":
                        # Implement zero padding
                        padding_size = item_len - x_item.shape[1]
                        x_item = torch.nn.functional.pad(x_item, (0, padding_size))
                    case _:
                        raise Exception(f"Padding method '{padding}' not implemented.")
            batch.append(x_item)
            batch_paths.append(output_path)

            if len(batch) == batch_size:
                batch = torch.stack(batch)
                batch = batch.to(device)
                with torch.no_grad():
                    embeddings = extractor(batch)
                save_or_append(embeddings.cpu(), batch_paths)
                batch = []
                batch_paths = []

        pbar.update(1)

    # Process the last batch
    if len(batch) > 0:
        while len(batch) < batch_size:
            batch.append(torch.zeros_like(batch[-1]))
            batch_paths.append(batch_paths[-1])
        batch = torch.stack(batch)
        batch = batch.to(device)
        with torch.no_grad():
            embeddings = extractor(batch)
        save_or_append(embeddings.cpu(), batch_paths)

    pbar.close()


def fixed_batch_extractor(
    dataset,
    extractor,
    batch_size: int = 32,
    device: str = "cpu",
):
    """
    Normal batch extractor for fixed length items
    (e.g. images from ImageNet that are transformed to a fixed size).
    """
    pbar = tqdm(total=len(dataset))
    extractor = extractor.to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    for i, (x, _) in enumerate(dataloader):
        x = x.to(device)
        with torch.no_grad():
            embeddings = extractor(x)

        # squeeze channels if present
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        for j, emb in enumerate(embeddings):
            emb = emb.unsqueeze(0)
            output_path = dataset.feature_paths[i * batch_size + j]
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(emb, path)

        pbar.update(batch_size)


class DynamicBatchSampler(Sampler):
    """
    Dynamic batch sampler for variable length items.

    Unlike the dynamic_batch_extractor, it doesn't rely on
    saving and identifying saved features for deciding
    how to batch items, so it can be used for training.
    However, it also needs to load the whole dataset once
    at the start to determine the total real length (items).

    Args:
        dataset: PyTorch dataset that returns [audio, label].
        batch_size: Batch size.
    """

    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        # !NOTE this will currently only work only if the dataset
        # returns items containing lists of arbitrarily many features
        # (arrays) of the same length (e.g. audio embeddings).
        self.item_lengths = [len(item[0]) for item in dataset]

        # subaray indexing, rather than dataset indexing
        self.real_len = sum(self.item_lengths)
        self.real_indices = list(range(self.real_len))

        # subarray indexing to (array, relative offset) tuple
        self.idx_map = {}
        real_idx = 0
        for i, item_len in enumerate(self.item_lengths):
            for j in range(item_len):
                self.idx_map[real_idx] = (i, j)
                real_idx += 1

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.real_indices)

        current_batch = []
        for real_idx in self.real_indices:
            array_idx, offset = self.idx_map[real_idx]
            current_batch.append((array_idx, offset))

            if len(current_batch) == self.batch_size:
                yield current_batch
                current_batch = []

        if current_batch:
            yield current_batch

    def __len__(self):
        return (self.real_len + self.batch_size - 1) // self.batch_size


def collate_packed_batch(batch, dataset):
    """
    Collate a batch of variable length items into a packed sequence.
    """
    sequences = []
    labels = []

    for array_idx, subarray_idx in batch:
        array, label = dataset[array_idx]
        subarray = array[subarray_idx]
        sequences.append(subarray)
        labels.append(label)

    packed_sequences = torch.cat(sequences)
    packed_labels = torch.tensor(labels)

    return packed_sequences, packed_labels


def compute_and_save_feature_stats(dataset_name, feature_name, label=None):
    """
    Compute and save the mean and standard deviation of features from .pt files.

    Args:
        dataset_name: Name of the dataset.
        feature_name: Name of the feature, used for naming the stats file.
        label: Optional label/concept name (for SynTheory dataset).
    """
    # Construct features directory path
    if label:
        features_dir = Path(f"data/{dataset_name}/{label}/{feature_name}")
    else:
        features_dir = Path(f"data/{dataset_name}/{feature_name}")
    
    all_features = []

    # Traverse the directory and load all .pt files
    for root, _, files in os.walk(features_dir):
        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                features = torch.load(file_path)
                if isinstance(features, dict):
                    # If the .pt file is a dictionary, extract the tensor
                    features = features.get('features', None)
                if features is not None:
                    all_features.append(features)

    # Concatenate all features into a single tensor
    all_features = torch.cat(all_features, dim=0)

    # Compute mean and std
    mean = all_features.mean().item()
    std = all_features.std().item()

    # Save the stats to a file
    stats_dir = Path("stats")
    stats_dir.mkdir(exist_ok=True)
    
    stats_filename = f"mean_std_{dataset_name}_{feature_name}.txt"
    
    stats_path = stats_dir / stats_filename
    with open(stats_path, 'w') as f:
        f.write(f"mean: {mean}\n")
        f.write(f"std: {std}\n")

    print(f"Feature stats saved to {stats_path}")

