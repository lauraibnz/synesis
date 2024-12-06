import importlib
from pathlib import Path
from config.features import feature_configs

import numpy as np
import torch
from torch.utils.data import Sampler
from tqdm import tqdm


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
        extract_kws = feature_configs[name].get("extract_kws", {})
        
        if kwargs:
            kwargs.update(extract_kws)
            
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
        weights_path = Path("models") / "pretrained" / f"{name.lower()}.pt"
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()

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

            if path.exists():
                emb_old = torch.load(path, weights_only=True)
                emb = torch.cat((emb_old, emb), dim=0)

            torch.save(emb, path)

    # Identify if .pt files are already present, and if so, skip them.
    # We do this now as the dynamic extractor writes and loads .pt files,
    # concatenating new embeddings, so we don't want to interefere with that.
    existing_files = [p for p in dataset.feature_paths if Path(p).exists()]

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
