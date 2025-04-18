import json
import os
import os.path as osp
import random

import numpy as np
import ray
import torch
import typer
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)
from torch_geometric.loader import DataLoader

from src.dataset.astroturf_graph_dataset import AstroturfCampaignGraphDataset
from src.helpers.device_helpers import get_device
from src.helpers.early_stopping import EarlyStopping
from src.modules.graph_encoder import UPFDGraphSageNet
from src.modules.loss.focal_loss import FocalLoss


def get_dataset_attributes(dataset):
    """Return the feature dimension and number of classes for a dataset."""
    first_graph = next(iter(dataset))
    feature_dim = first_graph.x.size(1)
    num_classes = 2
    return feature_dim, num_classes


def downsample_majority_class(dataset, majority_label: int = 0):
    majority_files, minority_files = [], []
    for filename in dataset._file_list:
        with open(osp.join(dataset.raw_dir, filename), "r") as json_file:
            label_str = json.load(json_file).get("label", "real").lower()
            label_int = 0 if label_str == "real" else 1
        (majority_files if label_int == majority_label else minority_files).append(filename)
    if not minority_files:
        raise ValueError("No minority‑class samples found!")
    dataset._file_list = random.sample(majority_files, len(minority_files)) + minority_files
    random.shuffle(dataset._file_list)
    return dataset


def train_one_epoch(
        model,
        data_loader,
        optimizer,
        device,
        criterion,
        max_grad_norm: float = 5.0,
):
    model.train()
    total_loss = total_correct = total_examples = 0
    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, *_ = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_correct += (logits.argmax(dim=-1) == batch.y).sum().item()
        total_examples += batch.num_graphs
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate_pr_auc(model, data_loader, device):
    model.eval()
    labels_true, probabilities = [], []
    for batch in data_loader:
        batch = batch.to(device)
        logits, *_ = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        probs = torch.nan_to_num(probs, nan=0.5).cpu().numpy()
        labels_true.extend(batch.y.cpu().numpy())
        probabilities.extend(probs)
    if np.isnan(probabilities).any():
        return 0.0
    return (
        average_precision_score(labels_true, probabilities)
        if len(set(labels_true)) > 1
        else 0.0
    )


@torch.no_grad()
def get_predictions(model, data_loader, device):
    labels_true, preds_hard = [], []
    model.eval()
    for batch in data_loader:
        batch = batch.to(device)
        logits, *_ = model(batch.x, batch.edge_index, batch.batch)
        labels_true.extend(batch.y.cpu().numpy())
        preds_hard.extend(logits.argmax(dim=-1).cpu().numpy())
    return labels_true, preds_hard


def run_single_train(
        train_dataset,
        val_dataset,
        test_dataset,
        hidden_channels,
        dropout,
        lr,
        weight_decay,
        epochs,
        batch_size,
        focal_alpha,
        focal_gamma,
        patience,
        save_model,
        save_dir,
):
    os.makedirs(save_dir, exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    in_channels, num_classes = get_dataset_attributes(train_dataset)
    device = get_device()
    model = UPFDGraphSageNet(in_channels, hidden_channels, num_classes, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    early_stopper = EarlyStopping(patience=patience, verbose=True, mode="max")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, criterion
        )
        val_pr = evaluate_pr_auc(model, val_loader, device)
        print(
            f"Epoch {epoch:03d} | loss {train_loss:.4f} | acc {train_acc:.4f} | val‑PR‑AUC {val_pr:.4f}"
        )
        early_stopper(val_pr, model)
        if early_stopper.early_stop:
            break
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    test_pr = evaluate_pr_auc(model, test_loader, device)
    y_true, y_pred = get_predictions(model, test_loader, device)
    print("\nClassification report")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix (rows true, cols pred):\n", confusion)
    if save_model:
        config = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "num_classes": num_classes,
        }
        torch.save(
            {"model_state_dict": model.state_dict(), "config": config},
            osp.join(save_dir, "graph_encoder.pth"),
        )
        torch.save({"confusion_matrix": confusion}, osp.join(save_dir, "confusion_matrix.pt"))
    return test_pr


def train_with_tune(config, max_epochs, train_dataset, val_dataset):
    in_channels, num_classes = get_dataset_attributes(train_dataset)
    device = get_device()
    model = UPFDGraphSageNet(
        in_channels,
        config["hidden_channels"],
        num_classes,
        config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    criterion = FocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    for _ in range(1, max_epochs + 1):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, *_ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
        val_pr = evaluate_pr_auc(model, val_loader, device)
        tune.report({"val_pr_auc": val_pr})


def hyperparam_search(max_epochs, n_samples, store, cfg_path, train_dataset, val_dataset):
    search_space = {
        "hidden_channels": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.0, 0.5),
        "lr": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([64, 128]),
        "focal_alpha": tune.uniform(0.1, 1.0),
        "focal_gamma": tune.uniform(0.5, 4.0),
    }
    scheduler = ASHAScheduler(
        metric="val_pr_auc", mode="max", max_t=max_epochs, grace_period=5
    )
    analysis = tune.run(
        tune.with_parameters(
            train_with_tune,
            max_epochs=max_epochs,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        ),
        config=search_space,
        num_samples=n_samples,
        scheduler=scheduler,
        storage_path="file://" + osp.abspath(store),
        resources_per_trial={"cpu": 2, "gpu": 1},
    )
    best_config = analysis.get_best_config(metric="val_pr_auc", mode="max")
    os.makedirs(osp.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as json_file:
        json.dump(best_config, json_file)
    return best_config


app = typer.Typer()


@app.command()
def main(
        dataset_root: str = typer.Option("../data/astroturf", "--dataset-root"),
        tuning: bool = typer.Option(False, "--tuning"),
        hidden_channels: int = typer.Option(64, "--hidden-channels"),
        dropout: float = typer.Option(0.3, "--dropout"),
        lr: float = typer.Option(1e-3, "--lr"),
        weight_decay: float = typer.Option(5e-4, "--weight-decay"),
        batch_size: int = typer.Option(128, "--batch-size"),
        epochs: int = typer.Option(30, "--epochs"),
        patience: int = typer.Option(10, "--patience"),
        tune_max_epochs: int = typer.Option(30, "--tune-max-epochs"),
        samples: int = typer.Option(10, "--samples"),
        model_output: str = typer.Option("../models/graph_encoder/", "--model-output-path"),
        downsample: bool = typer.Option(True, "--downsample"),
):
    root_dir = osp.abspath(osp.expanduser(dataset_root))
    save_dir = osp.abspath(osp.expanduser(model_output))
    train_dataset = AstroturfCampaignGraphDataset(root_dir, "train", shuffle=True)
    val_dataset = AstroturfCampaignGraphDataset(root_dir, "test")
    test_dataset = AstroturfCampaignGraphDataset(root_dir, "test")
    if downsample:
        train_dataset = downsample_majority_class(train_dataset)
    if tuning:
        ray.init(ignore_reinit_error=True)
        best_config = hyperparam_search(
            tune_max_epochs,
            samples,
            "./ray_results",
            osp.join(save_dir, "best_config.json"),
            train_dataset,
            val_dataset,
        )
        pr_auc = run_single_train(
            train_dataset,
            val_dataset,
            test_dataset,
            best_config["hidden_channels"],
            best_config["dropout"],
            best_config["lr"],
            best_config["weight_decay"],
            epochs,
            best_config["batch_size"],
            best_config["focal_alpha"],
            best_config["focal_gamma"],
            patience,
            True,
            save_dir,
        )
        ray.shutdown()
    else:
        pr_auc = run_single_train(
            train_dataset,
            val_dataset,
            test_dataset,
            hidden_channels,
            dropout,
            lr,
            weight_decay,
            epochs,
            batch_size,
            1.0,
            2.0,
            patience,
            True,
            save_dir,
        )
    print(f"\nFinal Test PR‑AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    app()
