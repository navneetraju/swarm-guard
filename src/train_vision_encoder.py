import json
import os
from datetime import datetime

import ray
import torch
import typer
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification
)

from src.dataset.astroturf_vision_dataset import AstroImageDataset, create_labeled_image_df
from src.helpers.early_stopping import EarlyStopping


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")

@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        logits = model(pixel_values=pixel_values).logits
        prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        ys.extend(labels.cpu().numpy())
        ps.extend(prob)
    try:
        return average_precision_score(ys, ps)
    except ValueError:
        return 0.0


def run_single_train(
        train_ds, val_ds, test_ds,
        model_id: str,
        lr: float, weight_decay: float,
        epochs: int, batch_size: int,
        patience: int,
        save_model: bool,
        model_out: str,
        write_tb: bool
):
    device = get_device()

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = ViTForImageClassification.from_pretrained(
        model_id,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, verbose=True, mode='max')

    writer = SummaryWriter(log_dir=f"./runs/vision/{datetime.now():%Y%m%d-%H%M%S}") if write_tb else None
    best_val_auc = -float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}/{epochs}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_auc = evaluate_auc(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Val Average Precision={val_auc:.4f}")


        if writer:
            writer.add_scalar("Train/Loss", avg_loss, epoch)
            writer.add_scalar("Val/Average_Precision", val_auc, epoch)

        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            print("🏁 Early stopping")
            break

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            model.save_pretrained(model_out)
            AutoImageProcessor.from_pretrained(model_id).save_pretrained(model_out)

    test_auc = evaluate_auc(model, test_loader, device)
    print(f"Test Average Precision: {test_auc:.4f}")

    preds, labels = [], []
    model.eval()
    for batch in tqdm(test_loader, desc="Final Classification Report"):
        pixel_values = batch["pixel_values"].to(device)
        logits = model(pixel_values=pixel_values).logits
        batch_preds = logits.argmax(dim=-1).cpu().numpy()
        preds.extend(batch_preds)
        labels.extend(batch["labels"].cpu().numpy())

    print(classification_report(labels, preds, target_names=["Real", "Fake"]))
    if writer:
        writer.close()
    return test_auc


def train_fn_tune(config, model_id, train_ds, val_ds, epochs: int = 5):
    device = get_device()

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = ViTForImageClassification.from_pretrained(
        model_id,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    for _ in tqdm(range(epochs), desc="Tune Training"):
        model.train()
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            loss = model(pixel_values=pixel_values, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    auc = evaluate_auc(model, val_loader, device)
    tune.report({"val_roc_auc": auc})



def hyperparam_search(model_id, train_ds, val_ds, num_samples: int, max_epochs: int, out_cfg: str):
    space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
    }
    sched = ASHAScheduler(metric="val_roc_auc", mode="max", max_t=max_epochs, grace_period=1)
    analysis = tune.run(
        tune.with_parameters(train_fn_tune, model_id=model_id, train_ds=train_ds, val_ds=val_ds),
        config=space,
        num_samples=num_samples,
        scheduler=sched,
        resources_per_trial={"cpu": 1, "gpu": 1},
        name="vision_search"
    )
    best = analysis.get_best_config("val_roc_auc", "max")
    print("Best tune config:", best)
    os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
    with open(out_cfg, "w") as f:
        json.dump(best, f)
    return best


app = typer.Typer()


@app.command()
def main(
        train_image_dir: str = typer.Option("path_to_bucket/swarm_guard_dataset/train/images",
                                            help="Path to training images"),
        train_json_dir: str = typer.Option("path_to_bucket/swarm_guard_dataset/train/json",
                                           help="Path to training JSONs"),
        test_image_dir: str = typer.Option("path_to_bucket/swarm_guard_dataset/test/images",
                                           help="Path to test images"),
        test_json_dir: str = typer.Option("path_to_bucket/swarm_guard_dataset/test/json", help="Path to test JSONs"),
        model_id: str = typer.Option("google/vit-base-patch16-224", help="HF model ID"),
        tuning: bool = typer.Option(False, help="Run hyperparam tuning?"),
        lr: float = typer.Option(1e-4),
        weight_decay: float = typer.Option(1e-5),
        epochs: int = typer.Option(5),
        batch_size: int = typer.Option(32),
        patience: int = typer.Option(3),
        tune_samples: int = typer.Option(5),
        tune_epochs: int = typer.Option(3),
        output_dir: str = typer.Option("./vision_model"),
):
    # Create datasets using your specific paths
    train_df = create_labeled_image_df(train_image_dir, train_json_dir)
    test_df = create_labeled_image_df(test_image_dir, test_json_dir)

    # Split train into train/val
    train_idx, val_idx = train_test_split(
        train_df.index,
        test_size=0.2,
        stratify=train_df["label"],
        random_state=42
    )

    # Create datasets
    full_train_ds = AstroImageDataset(train_df, model_id)
    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_train_ds, val_idx)
    test_ds = AstroImageDataset(test_df, model_id)

    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    if tuning:
        ray.init(ignore_reinit_error=True)
        cfg_path = os.path.join(output_dir, "best_vision_config.json")
        best_cfg = hyperparam_search(model_id, train_ds, val_ds, tune_samples, tune_epochs, cfg_path)
        final_auc = run_single_train(
            train_ds, val_ds, test_ds,
            model_id=model_id,
            lr=best_cfg["lr"],
            weight_decay=best_cfg["weight_decay"],
            epochs=epochs,
            batch_size=best_cfg["batch_size"],
            patience=patience,
            save_model=True,
            model_out=output_dir,
            write_tb=True
        )
        print(f"[TUNED] Test ROC AUC: {final_auc:.4f}")
        ray.shutdown()
    else:
        final_auc = run_single_train(
            train_ds, val_ds, test_ds,
            model_id=model_id,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            save_model=True,
            model_out=output_dir,
            write_tb=False
        )
        print(f"[NO TUNE] Test ROC AUC: {final_auc:.4f}")


if __name__ == "__main__":
    app()