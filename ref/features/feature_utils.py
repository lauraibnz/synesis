from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Sampler


def get_pretrained_model(model_name: str):
    match model_name:
        case "vggish-mtat":
            from ref.features.vggish import VGGish

            model = VGGish()
            model.load_state_dict(
                torch.load(Path("models") / "pretrained" / "vggish_mtat.pt")
            )
            model.eval()
        case _:
            raise ValueError(f"Invalid model name: {model_name}")
    return model


def dynamic_batch_extractor(
    dataset,
    extractor,
    item_len: int,
    padding: str = "repeat",
    batch_size: int = 32,
):
    """
    Create and process batches from a PyTorch dataset with items of
    variable length (such as audio tracks).

    Args:
        dataset: PyTorch dataset that returns [audio, label].
        item_len: Length of each eventual item (not item returned by dataset)
                  (in samples if audio).
        extractor: Function (or class with forward method) to process batches.
        padding: Padding method for items, either "repeat" or "zero".
        batch_size: Batch size.
    """

    def save_or_append(batch, batch_paths):
        for idx, output_path in enumerate(batch_paths):
            emb = batch[idx]
            if emb.shape[0] != 1:
                emb = np.expand_dims(emb, axis=0)

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists():
                with path.open("rb") as f:
                    emb_old = np.load(f)
                emb = np.concatenate((emb_old, emb), axis=0)

            with path.open("wb") as f:
                np.save(f, emb)

    pbar = tqdm(total=len(dataset))
    batch = []
    batch_paths = []

    for i in range(len(dataset)):
        x, _ = dataset[i]  # Ignore the label
        output_path = dataset.feature_paths[i]

        for buffer in range(0, len(x), item_len):
            x_item = x[buffer : buffer + item_len]
            if len(x_item) < item_len:
                match padding:
                    case "repeat":
                        x_item = x_item.repeat(int(np.ceil(item_len / len(x_item))))
                        x_item = x_item[:item_len]
                    case "zero":
                        x_item = torch.nn.functional.pad(
                            x_item,
                            (0, item_len - len(x_item)),
                            "constant",
                            0,
                        )
                    case _:
                        raise Exception(f"Padding method '{padding}' not implemented.")
            batch.append(x_item)
            batch_paths.append(output_path)

            if len(batch) == batch_size:
                batch = torch.stack(batch)
                with torch.no_grad():
                    embeddings = extractor(batch)
                save_or_append(embeddings, batch_paths)
                batch = []
                batch_paths = []

        pbar.update(1)

    # Process the last batch
    if len(batch) > 0:
        while len(batch) < batch_size:
            batch.append(torch.zeros_like(batch[-1]))
            batch_paths.append(batch_paths[-1])
        batch = torch.stack(batch)
        with torch.no_grad():
            embeddings = extractor(batch)
        save_or_append(embeddings, batch_paths)

    pbar.close()


class DynamicBatchSampler(Sampler):
    """
    Dynamic batch sampler for variable length items.

    Unlike the dynamic_batch_extractor, it doesn't rely on
    saving and identifying saved features for deciding
    how to batch items, so it can be used for training.
    However, it also needs to load the whole dataset once
    at the start to determine the total length.

    Args:
        dataset: PyTorch dataset that returns [audio, label].
        batch_size: Batch size.
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.item_lengths = [len(item[0]) for item in dataset]

    def __iter__(self):
        current_batch = []
        current_length = 0
        indices = np.random.permutation(len(self.dataset))

        for idx in indices:
            item_length = self.item_lengths[idx]
            if current_length + item_length > self.batch_size:
                yield current_batch
                current_batch = [idx]
                current_length = item_length
            else:
                current_batch.append(idx)
                current_length += item_length

        if current_batch:
            yield current_batch

    def __len__(self):
        return (sum(self.item_lengths) + self.batch_size - 1) // self.batch_size


def collate_packed_batch(batch):
    """
    Collate a batch of variable length items into a packed sequence.
    """
    sequences = []
    labels = []
    # offsets = [0]

    for seq, label in batch:
        sequences.append(seq)
        labels.append(label)
        # offsets.append(offsets[-1] + len(seq))

    packed_sequences = torch.cat(sequences)
    packed_labels = torch.tensor(labels)
    # offsets = torch.tensor(offsets[:-1])

    return packed_sequences, packed_labels
