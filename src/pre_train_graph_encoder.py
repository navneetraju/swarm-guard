import json
import os
from datetime import datetime

import ray
import torch
import typer
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import classification_report
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from helpers.early_stopping import EarlyStopping
from modules.graph_encoder import UPFDGraphSageNet


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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


def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, _, _ = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * data.num_graphs
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
        total_examples += data.num_graphs

    avg_loss = total_loss / total_examples  # consistent loss averaging
    acc = total_correct / total_examples
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    for data in loader:
        data = data.to(device)
        out, _, _ = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)

        total_loss += float(loss) * data.num_graphs
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
        total_examples += data.num_graphs

    avg_loss = total_loss / total_examples
    acc = total_correct / total_examples
    return avg_loss, acc


@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Gather all predicted labels and ground truth labels from the loader.
    This is for computing classification metrics like precision/recall/F1.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        out, _, _ = model(data.x, data.edge_index, data.batch)
        preds = out.argmax(dim=-1).cpu().numpy()
        labels = data.y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    return all_labels, all_preds


def dataset_stats(dataset):
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


def create_artificial_imbalance(dataset, ratio=0.4):
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
    new_majority_count = max(new_majority_count, 1)  # ensure at least one sample

    selected_indices = torch.randperm(len(majority_indices))[:new_majority_count].tolist()
    selected_majority_indices = [majority_indices[i] for i in selected_indices]

    new_indices = sorted(minority_indices + selected_majority_indices)
    data_list = [dataset[i] for i in new_indices]
    new_data, new_slices = InMemoryDataset.collate(data_list)

    # Instead of creating a new instance (which requires a missing attribute),
    # we update the dataset in place.
    dataset.data = new_data
    dataset.slices = new_slices
    return dataset


def load_datasets(combine: bool = True, create_artificial_imbalance_flag: bool = True, imbalance_ratio: float = 0.05):
    """
    Helper to load Politifact (and optionally GossipCop) train/val/test with 'profile' features.
    Returns train/val/test PyG datasets.
    """
    if combine:
        poli_train = UPFD("../data", "politifact", "profile", 'train')
        poli_val = UPFD("../data", "politifact", "profile", 'val')
        poli_test = UPFD("../data", "politifact", "profile", 'test')

        if create_artificial_imbalance_flag:
            poli_train = create_artificial_imbalance(poli_train, imbalance_ratio)
            poli_val = create_artificial_imbalance(poli_val, imbalance_ratio)
            print("Training set stats:")
            dataset_stats(poli_train)
            print("Validation set stats:")
            dataset_stats(poli_val)
            print("Test set stats:")
            dataset_stats(poli_test)

        gossip_train = UPFD("../data", "gossipcop", "profile", 'train')
        gossip_val = UPFD("../data", "gossipcop", "profile", 'val')
        gossip_test = UPFD("../data", "gossipcop", "profile", 'test')

        if create_artificial_imbalance_flag:
            gossip_train = create_artificial_imbalance(gossip_train, imbalance_ratio)
            gossip_val = create_artificial_imbalance(gossip_val, imbalance_ratio)
            print("GossipCop training set stats:")
            dataset_stats(gossip_train)
            print("GossipCop validation set stats:")
            dataset_stats(gossip_val)
            print("GossipCop test set stats:")
            dataset_stats(gossip_test)

        train_dataset = poli_train + gossip_train
        val_dataset = poli_val + gossip_val
        test_dataset = poli_test + gossip_test
    else:
        train_dataset = UPFD("../data", "politifact", "profile", 'train')
        val_dataset = UPFD("../data", "politifact", "profile", 'val')
        test_dataset = UPFD("../data", "politifact", "profile", 'test')
        if create_artificial_imbalance_flag:
            train_dataset = create_artificial_imbalance(train_dataset, imbalance_ratio)
            val_dataset = create_artificial_imbalance(val_dataset, imbalance_ratio)
            dataset_stats(train_dataset)
            dataset_stats(val_dataset)
            dataset_stats(test_dataset)
    return train_dataset, val_dataset, test_dataset


def run_single_train(
        train_dataset,
        val_dataset,
        test_dataset,
        hidden_channels: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
        patience: int = 10,
        save_model: bool = False,
        saved_model_path: str = "graph_encoder.pth",
        write_to_tensorboard: bool = False,
        tensorboard_log_dir: str = "./runs"
):
    """
    A single training run using the provided hyperparameters.
    Returns final test accuracy.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if isinstance(train_dataset, ConcatDataset):
        in_channels = train_dataset.datasets[0].num_features
        num_classes = train_dataset.datasets[0].num_classes
    else:
        in_channels = train_dataset.num_features
        num_classes = train_dataset.num_classes

    device = get_device()

    model = UPFDGraphSageNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = compute_class_weights(train_dataset, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    early_stopper = EarlyStopping(patience=patience, verbose=True)

    writer = SummaryWriter(log_dir=tensorboard_log_dir) if write_to_tensorboard else None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if write_to_tensorboard:
            writer.add_scalars("Accuracy", {"Train": train_acc, "Val": val_acc}, epoch)
            writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)

        early_stopper(val_acc, model)
        if early_stopper.early_stop:
            model.load_state_dict(torch.load('best_model.pth'))
            break

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    y_true, y_pred = get_predictions(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=["Real", "Fake"])
    print("\nClassification Report:\n", report)

    with open(f"{tensorboard_log_dir}/classification_report.txt", "w") as f:
        f.write(report)

    early_stopper.clean_up()

    if write_to_tensorboard:
        writer.close()

    if save_model:
        torch.save(model.state_dict(), saved_model_path)

    return test_acc


def train_with_tune(config, max_epochs, val_dataset):
    """
    Train function for Ray Tune. config is a dict of hyperparams from the search space.
    We do a short run, then we report val_loss to Ray.
    """
    if isinstance(val_dataset, ConcatDataset):
        in_channels = val_dataset.datasets[0].num_features
        num_classes = val_dataset.datasets[0].num_classes
    else:
        in_channels = val_dataset.num_features
        num_classes = val_dataset.num_classes

    device = get_device()

    model = UPFDGraphSageNet(
        in_channels=in_channels,
        hidden_channels=config["hidden_channels"],
        num_classes=num_classes,
        dropout=config["dropout"]
    ).to(device)
    print(f"in_channels: {in_channels}, num_classes: {num_classes}")
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    weights = compute_class_weights(val_dataset, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    for epoch in range(1, max_epochs + 1):
        val_loss, _ = train_one_epoch(model, val_loader, optimizer, device, criterion)
        tune.report({"val_loss": val_loss})


def hyperparam_search(max_epochs: int, num_samples: int, local_dir: str, saved_config_path: str, val_dataset):
    """
    Defines a search space, runs Ray Tune, saves best config to a JSON file.
    """
    search_space = {
        "hidden_channels": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.0, 0.5),
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([64, 128, 256])
    }

    scheduler = ASHAScheduler(metric="val_loss", mode="min", max_t=max_epochs, grace_period=5, reduction_factor=2)
    storage_uri = "file://" + os.path.abspath(local_dir)

    analysis = tune.run(
        tune.with_parameters(train_with_tune, max_epochs=max_epochs, val_dataset=val_dataset),
        resources_per_trial={"cpu": 2, "gpu": 0},
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path=storage_uri
    )

    best_config = analysis.get_best_config(metric="val_loss", mode="min")
    print("Best config found:", best_config)

    if not os.path.exists(os.path.dirname(saved_config_path)):
        os.makedirs(os.path.dirname(saved_config_path))
    with open(saved_config_path, "w") as f:
        json.dump(best_config, f)

    return best_config


app = typer.Typer()


@app.command()
def main(
        tuning: bool = typer.Option(False, help="Whether to run Ray Tune hyperparam search"),
        hidden_channels: int = typer.Option(64, help="Hidden dimension for GNN if not tuning"),
        dropout: float = typer.Option(0.3, help="Dropout rate if not tuning"),
        lr: float = typer.Option(0.001, help="Learning rate if not tuning"),
        weight_decay: float = typer.Option(5e-4, help="Weight decay if not tuning"),
        batch_size: int = typer.Option(128, help="Batch size if not tuning"),
        combine: bool = typer.Option(True, help="Whether to combine Politifact + GossipCop"),
        epochs: int = typer.Option(30, help="Number of epochs for training/tuning"),
        model_output_path: str = typer.Option("../models/graph_encoder.pth", help="Where to save final model"),
):
    """
    If --tuning=False, run a single training with the provided hyperparams.
    If --tuning=True, do Ray Tune hyperparam search, store best config, then do a final train with that config.
    """
    train_dataset, val_dataset, test_dataset = load_datasets(combine)

    if tuning:
        tensorboard_log_dir = f"./runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ray.init(ignore_reinit_error=True)

        best_config = hyperparam_search(
            max_epochs=epochs,
            num_samples=10,
            local_dir="./ray_results",
            saved_config_path=f"{tensorboard_log_dir}/best_config.json",
            val_dataset=val_dataset
        )

        final_test_acc = run_single_train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            hidden_channels=best_config["hidden_channels"],
            dropout=best_config["dropout"],
            lr=best_config["lr"],
            weight_decay=best_config["weight_decay"],
            epochs=epochs,
            batch_size=best_config["batch_size"],
            save_model=True,
            saved_model_path=model_output_path,
            write_to_tensorboard=True,
            tensorboard_log_dir=tensorboard_log_dir
        )
        print(f"[TUNING] Final test acc with best config = {final_test_acc:.4f}")
        ray.shutdown()
    else:
        final_test_acc = run_single_train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            hidden_channels=hidden_channels,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            save_model=True,
            saved_model_path=model_output_path
        )
        print(f"[NO-TUNING] Final test acc = {final_test_acc:.4f}")


if __name__ == "__main__":
    app()
