import random

import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset

from src.dataset.astroturf_graph_dataset import AstroturfCampaignGraphDataset


def compute_class_weights(dataset, device):
    """
    Compute class weights for imbalanced datasets.
    For UPFD convention: label 0 is "Real" and label 1 is "Fake".
    The weight for each class is computed as:
        total_samples / (num_classes * count_in_class)
    """
    # If the dataset has a collated .data attribute, use it.
    if hasattr(dataset, 'data') and hasattr(dataset.data, 'y'):
        labels = dataset.data.y
        counts = labels.bincount()
        total = labels.size(0)
        # Use the dataset property num_classes.
        num_classes = dataset.num_classes if hasattr(dataset, "num_classes") else len(counts)
        weights = [total / (num_classes * count.item()) if count.item() > 0 else 1.0 for count in counts]
        return torch.tensor(weights, device=device, dtype=torch.float)
    else:
        # Fallback: iterate over each graph.
        all_labels = [data.y.item() for data in dataset]
        counts = np.bincount(all_labels)
        total = sum(counts)
        num_classes = len(counts)
        weights = [total / (num_classes * count) if count > 0 else 1.0 for count in counts]
        return torch.tensor(weights, device=device, dtype=torch.float)


def graph_dataset_stats(dataset):
    """
    Prints statistics for a torch_geometric dataset (or a Subset/ConcatDataset).
    This function iterates over the dataset so that any filtering (e.g., via Subset)
    is properly reflected.

    Statistics printed include:
      - Number of graphs
      - Feature dimension (from the first graph)
      - Total number of edges (summed over graphs)
      - Class distribution (counts for each label)
    """
    # If dataset is a Subset or ConcatDataset, iterate over its items.
    # For ConcatDataset, we assume a list-like behavior.
    if isinstance(dataset, (Subset, ConcatDataset)):
        data_list = [data for data in dataset]
    else:
        data_list = list(dataset)

    if len(data_list) == 0:
        print("Dataset is empty!")
        return

    num_graphs = len(data_list)

    # Determine feature dimension from the first sample (if available)
    first_sample = data_list[0]
    if hasattr(first_sample, "x") and first_sample.x is not None:
        # Handle the case where x might be a 1D tensor (e.g. a single feature per node)
        feat_dim = first_sample.x.size(1) if first_sample.x.ndim > 1 else 1
    else:
        feat_dim = None

    # Sum total number of edges over all graphs.
    total_edges = 0
    for data in data_list:
        if hasattr(data, "edge_index") and data.edge_index is not None:
            total_edges += data.edge_index.size(1)

    # Gather class labels from each graph (assuming data.y exists and is scalar).
    labels = []
    for data in data_list:
        if hasattr(data, "y") and data.y is not None:
            # In case data.y is a tensor, convert to Python scalar.
            labels.append(data.y.item() if torch.is_tensor(data.y) else data.y)

    # Compute class distribution
    class_distribution = {label: labels.count(label) for label in set(labels)}

    # Print dataset statistics
    print("Dataset statistics:")
    print(f"  Number of Graphs: {num_graphs}")
    if feat_dim is not None:
        print(f"  Feature Dimension: {feat_dim}")
    print(f"  Total Edges: {total_edges}")
    print("  Class Distribution:")
    for label, count in class_distribution.items():
        print(f"    Class {label}: {count} samples")


def create_artificial_imbalance_graph(dataset, ratio=0.4):
    """
    Create artificial imbalance in the dataset by undersampling the majority class.
    For UPFD convention, label 1 is "Fake" (assumed majority) and label 0 is "Real".

    If the dataset is a Subset, extract the underlying dataset and its indices.
    Then, create a new dataset (an instance of the same class) using InMemoryDataset.collate.
    """
    # Check if dataset is a Subset.
    if hasattr(dataset, "indices"):
        indices = dataset.indices
        full_dataset = dataset.dataset
    else:
        indices = list(range(len(dataset)))
        full_dataset = dataset

    majority_class = 1  # "Fake"
    minority_class = 0  # "Real"

    majority_indices = [i for i in indices if full_dataset[i].y.item() == majority_class]
    minority_indices = [i for i in indices if full_dataset[i].y.item() == minority_class]

    # Compute the new number of majority samples.
    new_majority_count = int(ratio * len(majority_indices))
    new_majority_count = max(new_majority_count, 1)
    selected_indices = torch.randperm(len(majority_indices))[:new_majority_count].tolist()
    selected_majority_indices = [majority_indices[i] for i in selected_indices]

    new_indices = sorted(minority_indices + selected_majority_indices)
    data_list = [full_dataset[i] for i in new_indices]

    new_data, new_slices = InMemoryDataset.collate(data_list)
    # Create a new dataset instance with the same class as full_dataset.
    new_dataset = full_dataset.__class__(root=full_dataset.root, split=full_dataset.split)
    new_dataset.data = new_data
    new_dataset.slices = new_slices
    return new_dataset


def load_astroturf_datasets(create_artificial_imbalance_flag: bool = False,
                            imbalance_ratio: float = 0.05,
                            root: str = "../data/astroturf"):
    """
    Load training, validation, and test datasets using AstroturfCampaignGraphDataset.
    The directory is expected to have subfolders 'train' and 'test'. A validation
    set is created by randomly splitting the training set (80/20 split).

    Parameters:
        create_artificial_imbalance_flag (bool): Whether to induce artificial imbalance.
        imbalance_ratio (float): Fraction of majority ("Fake") samples to retain.
        root (str): Root directory for the Astroturf dataset.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load the full training dataset.
    full_train_dataset = AstroturfCampaignGraphDataset(root=root, split='train')
    print("Total training samples:", len(full_train_dataset))

    # Create validation set via an 80/20 split.
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

    # Load test dataset.
    test_dataset = AstroturfCampaignGraphDataset(root=root, split='test')

    if create_artificial_imbalance_flag:
        print("Creating artificial imbalance with ratio:", imbalance_ratio)
        train_dataset = create_artificial_imbalance_graph(train_subset, imbalance_ratio)
        val_dataset = create_artificial_imbalance_graph(val_subset, imbalance_ratio)
    else:
        # If not applying imbalance, convert subsets to full datasets.
        # (This is optional—you might simply want to keep them as subsets.)
        train_dataset = full_train_dataset.__class__(root=root, split='train')
        train_dataset.data, train_dataset.slices = full_train_dataset.data, full_train_dataset.slices
        # Similarly for validation, we re-collate the subset samples.
        val_data_list = [full_train_dataset[i] for i in val_subset.indices]
        val_data, val_slices = InMemoryDataset.collate(val_data_list)
        val_dataset = full_train_dataset.__class__(root=root, split='train')
        val_dataset.data, val_dataset.slices = val_data, val_slices

    return train_dataset, val_dataset, test_dataset


def downsample_majority_class(dataset, majority_label=0):
    """
    Downsamples the majority class in a dataset.

    Args:
        dataset: A dataset where each sample has a .y attribute (expected to be a tensor with the label).
        majority_label: The label of the majority class. (Typically 0 for "Real" in your case.)

    Returns:
        A torch.utils.data.Subset containing all samples from the minority class and a
        random subset of the majority class samples equal in number to the minority class.
    """
    # Collect indices for majority and minority classes
    majority_indices = [i for i, data in enumerate(dataset) if data.y.item() == majority_label]
    minority_indices = [i for i, data in enumerate(dataset) if data.y.item() != majority_label]

    print(f"Before downsampling: {len(majority_indices)} majority samples, {len(minority_indices)} minority samples.")

    # Downsample majority to match the number of minority samples
    random.shuffle(majority_indices)
    majority_downsampled = majority_indices[:len(minority_indices)]

    # Combine the downsampled majority and all minority samples
    selected_indices = majority_downsampled + minority_indices
    random.shuffle(selected_indices)

    print(
        f"After downsampling: {len(majority_downsampled)} majority samples, {len(minority_indices)} minority samples, total: {len(selected_indices)} samples.")

    return Subset(dataset, selected_indices)
