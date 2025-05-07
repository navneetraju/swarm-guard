import json
import os
from datetime import datetime

import ray
import torch
import typer
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import classification_report, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

from src.modules.loss.focal_loss import FocalLoss
from src.dataset.astroturf_text_dataset import AstroturfTextDataset
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
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        ys.extend(batch['labels'].cpu().numpy())
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
    write_tb: bool,
    alpha: float,
    gamma: float
):
    device = get_device()
    collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(model_id))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    loss_fn = FocalLoss(alpha=alpha, gamma=gamma)

    writer = None
    if write_tb:
        tb_dir = f"./runs/text/{datetime.now():%Y%m%d-%H%M%S}"
        writer = SummaryWriter(log_dir=tb_dir)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = loss_fn(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        val_auc = evaluate_auc(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Val PR AUC={val_auc:.4f}")

        if writer:
            writer.add_scalar("Train/Loss", avg_loss, epoch)
            writer.add_scalar("Val/PR_AUC", val_auc, epoch)

        early_stopping(-val_auc, model)
        if early_stopping.early_stop:
            print("🏁 Early stopping")
            break

        if save_model:
            os.makedirs(model_out, exist_ok=True)
            model.save_pretrained(model_out)
            AutoTokenizer.from_pretrained(model_id).save_pretrained(model_out)

    test_auc = evaluate_auc(model, test_loader, device)
    print(f"Test PR AUC: {test_auc:.4f}")

    preds, labels = [], []
    model.eval()
    for batch in tqdm(test_loader, desc="Final Classification Report"):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.argmax(dim=-1).cpu().numpy()
        preds.extend(logits)
        labels.extend(batch['labels'].cpu().numpy())

    print(classification_report(labels, preds, target_names=["Real", "Fake"]))
    if writer:
        writer.close()
    return test_auc


def train_fn_tune(config, model_id, train_ds, val_ds, epochs: int = 5):
    loss_fn = FocalLoss(alpha=config["alpha"], gamma=config["gamma"])
    device = get_device()
    collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(model_id))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    for epoch in tqdm(range(1, epochs + 1), desc="Tune Training Epochs"):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = loss_fn(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    auc = evaluate_auc(model, val_loader, device)
    tune.report({"val_roc_auc": auc})


def hyperparam_search(model_id, train_ds, val_ds, num_samples: int, max_epochs: int, out_cfg: str):
    space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "alpha": tune.uniform(0.1, 2.0),
        "gamma": tune.uniform(0.0, 5.0)
    }
    sched = ASHAScheduler(metric="val_roc_auc", mode="max", max_t=max_epochs, grace_period=1)
    analysis = tune.run(
        tune.with_parameters(train_fn_tune, model_id=model_id, train_ds=train_ds, val_ds=val_ds),
        config=space,
        num_samples=num_samples,
        scheduler=sched,
        resources_per_trial={"cpu": 1, "gpu": 1},
        name="text_search"
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
    dataset_root: str = typer.Option("../dataset", help="Root folder with train/ and test/ subfolders"),
    model_id: str = typer.Option("answerdotai/ModernBERT-base", help="HF model ID"),
    tuning: bool = typer.Option(False, help="Run hyperparam tuning?"),
    lr: float = typer.Option(1e-4),
    weight_decay: float = typer.Option(1e-5),
    epochs: int = typer.Option(5),
    batch_size: int = typer.Option(32),
    patience: int = typer.Option(3),
    tune_samples: int = typer.Option(5),
    tune_epochs: int = typer.Option(3),
    alpha: float = typer.Option(1.0, help="Focal loss alpha"),
    gamma: float = typer.Option(2.0, help="Focal loss gamma"),
    output_dir: str = typer.Option("./text_model")
):
    train_path = os.path.join(dataset_root, "train", "graphs")
    test_path = os.path.join(dataset_root, "test", "graphs")

    assert os.path.exists(train_path), f"Missing train path: {train_path}"
    assert os.path.exists(test_path), f"Missing test path: {test_path}"

    full_train = AstroturfTextDataset(train_path, model_id)
    idxs = list(range(len(full_train)))
    labels = full_train.labels()
    train_idx, val_idx = train_test_split(
        idxs, test_size=0.2, stratify=labels, random_state=42
    )
    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_train, val_idx)
    test_ds = AstroturfTextDataset(test_path, model_id)

    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    if tuning:
        ray.init(ignore_reinit_error=True)
        cfg_path = os.path.join(output_dir, "best_text_config.json")
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
            write_tb=True,
            alpha=best_cfg["alpha"],
            gamma=best_cfg["gamma"],
        )
        print(f"[TUNED] Test PR AUC: {final_auc:.4f}")
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
            write_tb=False,
            alpha=alpha,
            gamma=gamma
        )
        print(f"[NO TUNE] Test PR AUC: {final_auc:.4f}")

if __name__ == "__main__":
    app()