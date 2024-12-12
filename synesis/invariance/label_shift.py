"""Methods for evaluating downstream model robustness to label shift.

!NOTE Currently only supports single-label classification tasks.
!NOTE Currently for variable-item-length dataset, consider the
whole item as a single sample to calculate class distribution,
rather than individual, constant-length items.
"""

from typing import Optional

import numpy as np
import torch
from scipy.stats import entropy
from sklearn.utils import shuffle

from config.tasks import task_configs
from synesis.utils import deep_update


def create_balanced_subsample(dataset, indices_by_class, target_size, seed=42):
    """
    Creates a class-balanced subsample of specified size.

    Args:
        dataset: Dataset object with labels and data
        indices_by_class: List of lists with item indices belonging
                          to each class
        target_size: Number of samples to include in the subsample
        seed: Random seed
    Returns:
        List of indices for the subsample
    """
    n_classes = len(indices_by_class)
    samples_per_class = target_size // n_classes

    all_indices = []
    for class_indices in indices_by_class:
        selected = np.random.choice(
            class_indices, size=samples_per_class, replace=False, seed=seed
        )
        all_indices.extend(selected)

    return shuffle(all_indices)


def create_shifted_subsample(
    dataset, target_distribution, indices_by_class, target_size, seed=42
):
    """
    Creates a subsample with specified class distribution without replacement.

    Args:
        dataset: Dataset object with labels and data
        target_distribution: Desired class distribution for the subsample
        indices_by_class: List of lists with item indices belonging
                          to each class
        target_size: Number of samples to include in the subsample
        seed: Random seed
    """
    all_indices = []

    # Calculate number of samples needed for each class
    samples_per_class = (target_distribution * target_size).astype(int)

    # Ensure we maintain total number of samples by adjusting last class
    samples_per_class[-1] = target_size - samples_per_class[:-1].sum()

    # Subsample indices for each class
    for class_idx, n_samples in enumerate(samples_per_class):
        # Ensure we don't try to sample more than available
        available = len(indices_by_class[class_idx])
        if n_samples > available:
            raise ValueError(
                f"Not enough samples in class {class_idx}. "
                f"Needed {n_samples}, but only {available} available."
            )

        selected_indices = np.random.choice(
            indices_by_class[class_idx], size=n_samples, replace=False, seed=seed
        )
        all_indices.extend(selected_indices)

    return shuffle(all_indices)


def get_controlled_shifts(original_distribution, n_steps, max_kl, seed=42):
    """
    Generate random distributions with controlled KL divergence steps.
    We do this by generating random distributions and accepting them
    if their KL divergence from the original distribution is close
    enough to the target KL divergence.

    Args:
        original_distribution: Array with class distribution
        n_steps: Number of KL divergence steps
        max_kl: Maximum KL divergence between distributions
        seed: Random seed
    Returns:
        List of distributions, one for each KL step
    """
    np.random.seed(seed)
    n_classes = len(original_distribution)

    # Create target KL divergences
    target_kls = np.linspace(0, max_kl, n_steps)
    shifted_distributions = []

    for target_kl in target_kls:
        found = False
        attempts = 0
        max_attempts = 1000  # Avoid infinite loops

        while not found and attempts < max_attempts:
            # Generate random distribution
            dist = np.random.dirichlet(np.ones(n_classes))

            # Calculate KL divergence
            kl = entropy(original_distribution, dist)

            # Accept if KL is close enough to target
            # Tolerance becomes larger as KL increases to maintain feasibility
            tolerance = max(0.1 * target_kl, 0.01)
            if abs(kl - target_kl) <= tolerance:
                shifted_distributions.append(dist)
                found = True

            attempts += 1

        if not found:
            print(
                f"Warning: Could not generate distribution for KL step {target_kl:.2f}"
            )

    return shifted_distributions


def create_shifted_datasets(dataset, n_steps, max_kl=2.0, seed=42):
    """
    Creates multiple subsampled versions of dataset with controlled label shift.

    NOTE! Currently, the subset size is determined by the number of examples in
    the smallest class, to ensure enough data is available for all classes.
    However, that might be seriously limiting in some cases.

    Args:
        dataset: Dataset object with labels and data
        n_steps: Number of steps to take from original distribution
        max_kl: Maximum KL divergence between original and shifted distributions
        seed: Random seed
    Returns:
        List of lists of indices for each shifted dataset,
        List of KL divergences between original and shifted distributions
    """
    # Calculate original distribution
    num_classes = len(dataset.labels[0])
    original_distribution = np.zeros(num_classes)  # num of items per class
    indices_by_class = [[] for _ in range(num_classes)]  # item idxs per class

    for idx, label in enumerate(dataset.labels):
        # labels are one-hot encoded, so we can find the class index by argmax
        class_idx = np.argmax(label)
        original_distribution[class_idx] += 1
        indices_by_class[class_idx].append(idx)

    original_distribution = original_distribution / len(dataset.labels)

    # Determine subsample size that ensures we can create all shifted distributions
    min_class_size = min(len(indices) for indices in indices_by_class)
    subsample_size = min_class_size * num_classes

    # Create balanced subsample of original distribution
    balanced_indices = create_balanced_subsample(
        dataset, indices_by_class, subsample_size, seed=seed
    )

    # Get target distributions with controlled KL steps
    target_distributions = get_controlled_shifts(
        original_distribution, n_steps, max_kl, seed=seed
    )

    # Create datasets with these distributions
    shifted_datasets = [balanced_indices]  # Start with balanced subsample
    kl_divergences = [0.0]  # KL divergence of original distribution with itself

    for target_dist in target_distributions:
        try:
            # in this, seed is used to choose item indices, so has nothing
            # to do with the class distribution
            indices = create_shifted_subsample(
                dataset, target_dist, indices_by_class, subsample_size, seed=seed
            )
            shifted_datasets.append(indices)
            kl = entropy(original_distribution, target_dist)
            kl_divergences.append(kl)
        except ValueError as e:
            print(f"Stopping at KL={kl}: {str(e)}")
            break

    return shifted_datasets, kl_divergences


def train(
    feature: str,
    dataset: str,
    task: str,
    item_format: str = "feature",
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
    seed: int = 42,
):
    """
    Train downstream models with subsets of the dataset with
    increasing KL divergence from a balanced distribution.

    Args:
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        task: Name of the downstream task (needs to be supported by dataset).
        item_format: Format of the input data: ["raw", "feature"].
                     Defaults to "feature". If raw, feature is
                     extracted on-the-fly.
        task_config: Override certain values of the task configuration.
        device: Device to use for training (defaults to "cuda" if available).
        seed: Random seed to use for subset distribution and item selection.

    Returns:
        List of models trained on subsets with increasing KL divergence,
        List of KL divergences between balanced and shifted distributions.
    """
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if task_config:
        task_configs[task] = deep_update(task_configs[task], task_config)

    pass
