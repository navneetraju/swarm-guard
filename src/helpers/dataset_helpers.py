import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import UPFD


def compute_class_weights(dataset, device):
    if hasattr(dataset, 'data') and hasattr(dataset.data, 'y'):
        labels = dataset.data.y
        counts = labels.bincount()
        total = labels.size(0)
        num_classes = len(counts)
        weights = [total / (num_classes * count.item()) if count.item() > 0 else 1.0 for count in counts]
        return torch.tensor(weights, device=device, dtype=torch.float)
    else:
        all_labels = [data.y.item() for data in dataset]
        import numpy as np
        counts = np.bincount(all_labels)
        total = sum(counts)
        num_classes = len(counts)
        weights = [total / (num_classes * count) if count > 0 else 1.0 for count in counts]
        return torch.tensor(weights, device=device, dtype=torch.float)


def graph_dataset_stats(dataset):
    """
    Print stylized dataset stats: number of classes, features, graphs, and edges.
    """
    divider = "=" * 50
    header = f" DATASET: {dataset.__class__.__name__} "
    print(divider)
    print(header.center(50, "="))
    print(divider)
    print(f"Number of Classes  : {dataset.num_classes}")
    print(f"Number of Features : {dataset.num_features}")
    print(f"Number of Graphs   : {len(dataset)}")
    print(f"Number of Edges    : {dataset.data.edge_index.size(1)}")
    print(divider)
    print("Class Distribution:")
    class_counts = dataset.data.y.bincount()
    for i, count in enumerate(class_counts):
        class_label = "Real" if i == 0 else "Fake"
        print(f"  Class {class_label:<4}: {count.item()} samples")
    print(divider, "\n")


def create_artificial_imbalance_graph(dataset, ratio=0.4):
    """
    Create artificial imbalance in the dataset by removing a percentage of the majority class.
    This is useful for testing the model's performance on imbalanced datasets.
    Majority class is assumed to be labeled as 1 and minority as 0.
    """
    indices = list(range(len(dataset)))
    majority_class = 1
    minority_class = 0

    majority_indices = [i for i in indices if dataset[i].y.item() == majority_class]
    minority_indices = [i for i in indices if dataset[i].y.item() == minority_class]

    new_majority_count = int(ratio * len(majority_indices))
    new_majority_count = max(new_majority_count, 1)

    selected_indices = torch.randperm(len(majority_indices))[:new_majority_count].tolist()
    selected_majority_indices = [majority_indices[i] for i in selected_indices]

    new_indices = sorted(minority_indices + selected_majority_indices)
    data_list = [dataset[i] for i in new_indices]
    new_data, new_slices = InMemoryDataset.collate(data_list)

    dataset.data = new_data
    dataset.slices = new_slices
    return dataset


def load_upfd_datasets(combine: bool = True, create_artificial_imbalance_flag: bool = True,
                       imbalance_ratio: float = 0.05):
    """
    Helper to load Politifact (and optionally GossipCop) train/val/test with 'profile' features.
    Returns train/val/test PyG datasets.
    """
    if combine:
        poli_train = UPFD("../data", "politifact", "profile", 'train')
        poli_val = UPFD("../data", "politifact", "profile", 'val')
        poli_test = UPFD("../data", "politifact", "profile", 'test')

        if create_artificial_imbalance_flag:
            poli_train = create_artificial_imbalance_graph(poli_train, imbalance_ratio)
            poli_val = create_artificial_imbalance_graph(poli_val, imbalance_ratio)
            print("Training set stats:")
            graph_dataset_stats(poli_train)
            print("Validation set stats:")
            graph_dataset_stats(poli_val)
            print("Test set stats:")
            graph_dataset_stats(poli_test)

        gossip_train = UPFD("../data", "gossipcop", "profile", 'train')
        gossip_val = UPFD("../data", "gossipcop", "profile", 'val')
        gossip_test = UPFD("../data", "gossipcop", "profile", 'test')

        if create_artificial_imbalance_flag:
            gossip_train = create_artificial_imbalance_graph(gossip_train, imbalance_ratio)
            gossip_val = create_artificial_imbalance_graph(gossip_val, imbalance_ratio)
            print("GossipCop training set stats:")
            graph_dataset_stats(gossip_train)
            print("GossipCop validation set stats:")
            graph_dataset_stats(gossip_val)
            print("GossipCop test set stats:")
            graph_dataset_stats(gossip_test)

        train_dataset = poli_train + gossip_train
        val_dataset = poli_val + gossip_val
        test_dataset = poli_test + gossip_test
    else:
        train_dataset = UPFD("../data", "politifact", "profile", 'train')
        val_dataset = UPFD("../data", "politifact", "profile", 'val')
        test_dataset = UPFD("../data", "politifact", "profile", 'test')
        if create_artificial_imbalance_flag:
            train_dataset = create_artificial_imbalance_graph(train_dataset, imbalance_ratio)
            val_dataset = create_artificial_imbalance_graph(val_dataset, imbalance_ratio)
            graph_dataset_stats(train_dataset)
            graph_dataset_stats(val_dataset)
            graph_dataset_stats(test_dataset)
    return train_dataset, val_dataset, test_dataset
