"""Methods for evaluating downstream model robustness
to label shift."""

import numpy as np
from scipy.stats import entropy
from sklearn.utils import shuffle


def create_balanced_subsample(dataset, indices_by_class, target_size):
    """Creates a class-balanced subsample of specified size."""
    n_classes = len(indices_by_class)
    samples_per_class = target_size // n_classes

    all_indices = []
    for class_indices in indices_by_class:
        selected = np.random.choice(
            class_indices, size=samples_per_class, replace=False
        )
        all_indices.extend(selected)

    return shuffle(all_indices)


def create_shifted_subsample(
    dataset, target_distribution, indices_by_class, target_size
):
    """Creates a subsample with specified class distribution without replacement."""
    all_indices = []

    # Calculate number of samples needed for each class
    samples_per_class = (target_distribution * target_size).astype(int)

    # Ensure we maintain total number of samples
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
            indices_by_class[class_idx], size=n_samples, replace=False
        )
        all_indices.extend(selected_indices)

    return shuffle(all_indices)


def get_controlled_shifts(original_distribution, n_steps, max_kl):
    """Generate a series of distributions with approximately constant KL steps."""
    n_classes = len(original_distribution)

    # Create increasingly imbalanced distributions
    shifted_distributions = []
    # current_kl = 0

    alpha_values = np.linspace(0, 1, n_steps)

    for alpha in alpha_values:
        shifted_dist = original_distribution.copy()
        dominant_class = np.argmax(original_distribution)

        # Increase probability of dominant class
        shifted_dist[dominant_class] = (1 - alpha) * original_distribution[
            dominant_class
        ] + alpha

        # Decrease others proportionally
        others = np.arange(n_classes) != dominant_class
        shifted_dist[others] *= (1 - shifted_dist[dominant_class]) / shifted_dist[
            others
        ].sum()

        # Calculate KL divergence
        kl = entropy(original_distribution, shifted_dist)

        if kl > max_kl:
            break

        shifted_distributions.append(shifted_dist)
        # current_kl = kl

    return shifted_distributions


def create_shifted_datasets(dataset, n_steps, subsample_size=None, max_kl=2.0):
    """Creates multiple subsampled versions of dataset with controlled label shift."""
    # Get original distribution
    labels = [label for _, label in dataset]
    unique_labels = np.unique(labels)

    # Calculate original distribution
    original_distribution = np.zeros(len(unique_labels))
    indices_by_class = [[] for _ in range(len(unique_labels))]

    for idx, label in enumerate(labels):
        original_distribution[label] += 1
        indices_by_class[label].append(idx)

    original_distribution = original_distribution / len(labels)

    # Determine subsample size if not provided
    if subsample_size is None:
        # Use size that ensures we can create all shifted distributions
        min_class_size = min(len(indices) for indices in indices_by_class)
        subsample_size = min_class_size * len(unique_labels)

    # Create balanced subsample of original distribution
    balanced_indices = create_balanced_subsample(
        dataset, indices_by_class, subsample_size
    )

    # Get target distributions with controlled KL steps
    target_distributions = get_controlled_shifts(original_distribution, n_steps, max_kl)

    # Create datasets with these distributions
    shifted_datasets = [balanced_indices]  # Start with balanced subsample
    kl_divergences = [0.0]  # KL divergence of original distribution with itself

    for target_dist in target_distributions:
        try:
            indices = create_shifted_subsample(
                dataset, target_dist, indices_by_class, subsample_size
            )
            shifted_datasets.append(indices)
            kl = entropy(original_distribution, target_dist)
            kl_divergences.append(kl)
        except ValueError as e:
            print(f"Stopping at KL={kl}: {str(e)}")
            break

    return shifted_datasets, kl_divergences
