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
from torch_geometric.loader import DataLoader

from helpers.dataset_helpers import compute_class_weights, load_upfd_datasets
from helpers.device_helpers import get_device
from helpers.early_stopping import EarlyStopping
from modules.graph_encoder import UPFDGraphSageNet


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
        saved_model_path: str = "../models/graph/graph_encoder.pth",
        write_to_tensorboard: bool = False,
        tensorboard_log_dir: str = "./runs",
        include_config: bool = False  # Flag to include model config when saving
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
        # Ensure that the model directory exists.
        model_dir = os.path.dirname(saved_model_path)
        os.makedirs(model_dir, exist_ok=True)

        if include_config:
            # Save a dictionary with both the model state and the configuration.
            model_config = {
                "in_channels": in_channels,
                "hidden_channels": hidden_channels,
                "out_channels": num_classes,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": epochs
            }
            torch.save({"model_state_dict": model.state_dict(), "config": model_config}, saved_model_path)
            # Save the config file in the same folder.
            config_file = os.path.join(model_dir, "graph_encoder_config.json")
            with open(config_file, "w") as f:
                json.dump(model_config, f)
        else:
            torch.save(model.state_dict(), saved_model_path)

    return test_acc


def train_with_tune(config, max_epochs, val_dataset):
    """
    Train function for Ray Tune. Config is a dict of hyperparameters from the search space.
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

    # Augment best_config with dataset-derived parameters.
    if isinstance(val_dataset, ConcatDataset):
        best_config["in_channels"] = val_dataset.datasets[0].num_features
        best_config["out_channels"] = val_dataset.datasets[0].num_classes
    else:
        best_config["in_channels"] = val_dataset.num_features
        best_config["out_channels"] = val_dataset.num_classes

    # Save the best config in the designated path.
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
        # Default model output path now in ../models/graph subfolder.
        model_output_path: str = typer.Option("../models/graph/graph_encoder.pth", help="Where to save final model"),
):
    """
    If --tuning=False, run a single training with the provided hyperparameters.
    If --tuning=True, do Ray Tune hyperparam search, store best config, then do a final train with that config.
    """
    train_dataset, val_dataset, test_dataset = load_upfd_datasets(combine)

    if tuning:
        # Define model directory based on model_output_path
        model_dir = os.path.dirname(model_output_path)
        os.makedirs(model_dir, exist_ok=True)
        # Save best config JSON in the same folder as model.
        saved_config_path = os.path.join(model_dir, "best_config.json")
        tensorboard_log_dir = f"./runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ray.init(ignore_reinit_error=True)

        best_config = hyperparam_search(
            max_epochs=epochs,
            num_samples=10,
            local_dir="./ray_results",
            saved_config_path=saved_config_path,
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
            saved_model_path=model_output_path,
            include_config=True  # Save both state dict and model config
        )
        print(f"[NO-TUNING] Final test acc = {final_test_acc:.4f}")


if __name__ == "__main__":
    app()
