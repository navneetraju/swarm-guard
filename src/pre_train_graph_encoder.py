import json
import os
import random
from datetime import datetime

import ray
import torch
import typer
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from src.helpers.dataset_helpers import compute_class_weights, load_astroturf_datasets, graph_dataset_stats
from src.helpers.device_helpers import get_device
from src.helpers.early_stopping import EarlyStopping
from src.modules.graph_encoder import UPFDGraphSageNet


# ---------------------------
# Helper Functions
# ---------------------------
def downsample_majority_class(dataset, majority_label=0):
    """
    Downsamples the majority class in a dataset.

    Args:
        dataset: A dataset where each sample has a .y attribute.
        majority_label: The label of the majority class.
    Returns:
        A torch.utils.data.Subset containing all minority class samples along with
        a random subset of the majority class equal in size to the minority class.
    """
    majority_indices = [i for i, data in enumerate(dataset) if data.y.item() == majority_label]
    minority_indices = [i for i, data in enumerate(dataset) if data.y.item() != majority_label]
    print(f"Before downsampling: {len(majority_indices)} majority samples, {len(minority_indices)} minority samples.")

    if not minority_indices:
        raise ValueError("No samples found for the minority class!")

    random.shuffle(majority_indices)
    majority_downsampled = majority_indices[:len(minority_indices)]
    selected_indices = majority_downsampled + minority_indices
    random.shuffle(selected_indices)
    print(
        f"After downsampling: {len(majority_downsampled)} majority samples, {len(minority_indices)} minority samples, total: {len(selected_indices)} samples.")

    return Subset(dataset, selected_indices)


def get_dataset_attributes(dataset):
    """
    Retrieve in_channels and num_classes from a dataset.
    Handles if the dataset is a Subset or a ConcatDataset.
    """
    if isinstance(dataset, ConcatDataset):
        base_ds = dataset.datasets[0]
    elif hasattr(dataset, 'dataset'):
        base_ds = dataset.dataset
    else:
        base_ds = dataset

    in_channels = getattr(base_ds, "num_features", base_ds[0].x.size(1))
    if hasattr(base_ds, "num_classes"):
        num_classes = base_ds.num_classes
    else:
        labels = [data.y.item() for data in base_ds]
        num_classes = int(max(labels)) + 1
    return in_channels, num_classes


# ---------------------------
# Training Functions
# ---------------------------
def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
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

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_auc(model, loader, device):
    """
    Evaluates the model using ROC AUC (binary classification, class 1 probability).
    """
    model.eval()
    y_true, y_prob = [], []
    for data in loader:
        data = data.to(device)
        out, _, _ = model(data.x, data.edge_index, data.batch)
        probs = torch.softmax(out, dim=-1)[:, 1].detach().cpu().numpy()
        y_true.extend(data.y.cpu().numpy())
        y_prob.extend(probs)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0
    return roc_auc


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out, _, _ = model(data.x, data.edge_index, data.batch)
        preds = out.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(data.y.cpu().numpy())
    return all_labels, all_preds


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
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        patience: int = 10,
        save_model: bool = False,
        saved_model_path: str = "../models/graph/graph_encoder.pth",
        write_to_tensorboard: bool = False,
        tensorboard_log_dir: str = "./runs",
        include_config: bool = False
):
    """
    Performs one training run with the provided hyperparameters.
    Uses ROC AUC for early stopping.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    in_channels, num_classes = get_dataset_attributes(train_dataset)

    device = get_device()
    model = UPFDGraphSageNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # By default, we are using a standard CrossEntropyLoss with class weights.
    weights = compute_class_weights(train_dataset, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    early_stopper = EarlyStopping(patience=patience, verbose=True, mode="max")
    writer = SummaryWriter(log_dir=tensorboard_log_dir) if write_to_tensorboard else None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_roc_auc = evaluate_auc(model, val_loader, device)
        print(
            f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val ROC AUC: {val_roc_auc:.4f}")

        if write_to_tensorboard:
            writer.add_scalar("Train Loss", train_loss, epoch)
            writer.add_scalar("Train Accuracy", train_acc, epoch)
            writer.add_scalar("Val ROC AUC", val_roc_auc, epoch)

        early_stopper(val_roc_auc, model)
        if early_stopper.early_stop:
            model.load_state_dict(torch.load('best_model.pth'))
            print("Early stopping triggered. Restoring best model...")
            break

    # Evaluate on test set
    test_roc_auc = evaluate_auc(model, test_loader, device)
    y_true, y_pred = get_predictions(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=["Real", "Fake"])
    print("\nClassification Report:\n", report)

    # Save classification report to disk
    report_path = os.path.join(tensorboard_log_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    early_stopper.clean_up()
    if write_to_tensorboard:
        writer.close()

    # Save the final model + config (if requested) to models/graph/
    if save_model:
        os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
        model_config = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "num_classes": num_classes,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma
        }
        if include_config:
            # Save both state dict + config in the .pth
            state = {
                "model_state_dict": model.state_dict(),
                "config": model_config
            }
            torch.save(state, saved_model_path)

            # Additionally, save config to "graph_encoder_config.json"
            config_path = os.path.join(os.path.dirname(saved_model_path), "graph_encoder_config.json")
            with open(config_path, "w") as f:
                json.dump(model_config, f)
        else:
            # Only save weights
            torch.save(model.state_dict(), saved_model_path)

    return test_roc_auc


def train_with_tune(config, max_epochs, val_dataset):
    """
    Train function for Ray Tune, reporting ROC AUC.
    """
    if hasattr(val_dataset, "dataset"):
        underlying = val_dataset.dataset
    else:
        underlying = val_dataset

    in_channels = getattr(underlying, "num_features", underlying[0].x.size(1))
    if hasattr(underlying, "num_classes"):
        num_classes = underlying.num_classes
    else:
        labels = [data.y.item() for data in underlying]
        num_classes = int(max(labels)) + 1

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
        # Dummy train logic for Ray Tune
        val_roc_auc = evaluate_auc(model, val_loader, device)
        tune.report({"val_roc_auc": val_roc_auc})


def hyperparam_search(max_epochs: int, num_samples: int, local_dir: str, saved_config_path: str, val_dataset):
    """
    Defines the search space and runs Ray Tune with Focal Loss hyperparams (focal_alpha, focal_gamma).
    """
    search_space = {
        "hidden_channels": tune.choice([64, 128, 256, 512, 1024]),
        "dropout": tune.uniform(0.0, 0.6),
        "lr": tune.loguniform(1e-4, 9e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "focal_alpha": tune.uniform(0.1, 1.0),
        "focal_gamma": tune.uniform(0.5, 5.0)
    }

    scheduler = ASHAScheduler(metric="val_roc_auc", mode="max", max_t=max_epochs, grace_period=5, reduction_factor=2)
    storage_uri = "file://" + os.path.abspath(local_dir)

    analysis = tune.run(
        tune.with_parameters(train_with_tune, max_epochs=max_epochs, val_dataset=val_dataset),
        resources_per_trial={"cpu": 2, "gpu": 0},
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path=storage_uri,
        verbose=1
    )

    best_config = analysis.get_best_config(metric="val_roc_auc", mode="max")
    print("Best config found:", best_config)

    # Retrieve dataset features + classes
    if hasattr(val_dataset, "dataset"):
        underlying = val_dataset.dataset
    else:
        underlying = val_dataset
    in_channels = getattr(underlying, "num_features", underlying[0].x.size(1))
    if hasattr(underlying, "num_classes"):
        num_classes = underlying.num_classes
    else:
        labels = [data.y.item() for data in underlying]
        num_classes = int(max(labels)) + 1

    # Add them for completeness
    best_config["in_channels"] = in_channels
    best_config["num_classes"] = num_classes

    os.makedirs(os.path.dirname(saved_config_path), exist_ok=True)
    with open(saved_config_path, "w") as f:
        json.dump(best_config, f)

    return best_config


app = typer.Typer()


@app.command()
def main(
        dataset_root: str = typer.Option("../data/astroturf", help="Root folder for the dataset"),
        tuning: bool = typer.Option(False, help="Whether to run hyperparameter tuning via Ray Tune"),
        hidden_channels: int = typer.Option(64, help="Hidden channels (if not tuning)"),
        dropout: float = typer.Option(0.3, help="Dropout rate (if not tuning)"),
        lr: float = typer.Option(0.001, help="Learning rate (if not tuning)"),
        weight_decay: float = typer.Option(5e-4, help="Weight decay (if not tuning)"),
        batch_size: int = typer.Option(128, help="Batch size (if not tuning)"),
        epochs: int = typer.Option(30, help="Number of epochs to train/tune"),
        model_output_path: str = typer.Option("../models/graph/graph_encoder.pth", help="Where to save the model"),
        downsample: bool = typer.Option(True, help="Whether to downsample the majority class in the training set")
):
    """
    Main entry point for training or hyperparameter tuning the graph encoder.

    If --tuning is used, Ray Tune hyperparameter search is performed.
    Otherwise, a single training run is done with the specified hyperparams.

    If --downsample is used, the majority class is reduced in size to match
    the minority class in the training set.
    """
    # Load the raw datasets
    train_dataset, val_dataset, test_dataset = load_astroturf_datasets(root=dataset_root)

    # Optional downsampling
    if downsample:
        train_dataset = downsample_majority_class(train_dataset, majority_label=0)

    print("Dataset Loaded:")
    print(f" - Train: {len(train_dataset)} samples")
    graph_dataset_stats(train_dataset)
    print(f" - Val: {len(val_dataset)} samples")
    graph_dataset_stats(val_dataset)
    print(f" - Test: {len(test_dataset)} samples")
    graph_dataset_stats(test_dataset)

    # Perform tuning or direct training
    if tuning:
        model_dir = os.path.dirname(model_output_path)
        os.makedirs(model_dir, exist_ok=True)
        saved_config_path = os.path.join(model_dir, "best_config.json")
        tensorboard_log_dir = f"./runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ray.init(ignore_reinit_error=True)

        # Hyperparam search
        best_config = hyperparam_search(
            max_epochs=epochs,
            num_samples=10,
            local_dir="./ray_results",
            saved_config_path=saved_config_path,
            val_dataset=val_dataset
        )

        # Final train using best hyperparams
        final_test_roc_auc = run_single_train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            hidden_channels=best_config["hidden_channels"],
            dropout=best_config["dropout"],
            lr=best_config["lr"],
            weight_decay=best_config["weight_decay"],
            focal_alpha=best_config.get("focal_alpha", 1.0),
            focal_gamma=best_config.get("focal_gamma", 2.0),
            epochs=epochs,
            batch_size=best_config["batch_size"],
            save_model=True,
            saved_model_path=model_output_path,
            write_to_tensorboard=True,
            tensorboard_log_dir=tensorboard_log_dir
        )
        print(f"[TUNING] Final test ROC AUC with best config = {final_test_roc_auc:.4f}")
        ray.shutdown()
    else:
        # Single training run with the specified hyperparameters
        final_test_roc_auc = run_single_train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            hidden_channels=hidden_channels,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            focal_alpha=1.0,  # Default focal alpha
            focal_gamma=2.0,  # Default focal gamma
            save_model=True,
            saved_model_path=model_output_path,
            include_config=True  # Save JSON config along with .pth
        )
        print(f"[NO-TUNING] Final test ROC AUC = {final_test_roc_auc:.4f}")


if __name__ == "__main__":
    app()
