import random
import tempfile
from pathlib import Path

import ray.cloudpickle as pickle
import torch
import typer
from ray import tune
from ray.tune import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from transformers import AutoModel

from src.dataset import AstroturfCampaignMultiModalDataset, astrorag_collate_fn
from src.helpers.device_helpers import move_to_device
from src.helpers.model_loaders import load_pre_trained_graph_encoder
from src.modules.loss.focal_loss import FocalLoss
from src.modules.multi_modal.multi_modal_model import MultiModalModelForClassification

app = typer.Typer()

search_space = {
    "self_attention_heads": tune.choice([1, 2, 4, 8, 16, 32]),
    "num_cross_modal_attention_heads": tune.choice([1, 2, 4, 8, 16, 32]),
    "embedding_dim": tune.sample_from(lambda spec: random.choice(
        [dim for dim in [32, 64, 128, 256, 512, 728, 1024]
         if dim % spec.config["self_attention_heads"] == 0
         and dim % spec.config["num_cross_modal_attention_heads"] == 0
         ]
    )),
    "self_attn_ff_dim": tune.choice([64, 128, 256, 512, 728, 1024]),
    "num_cross_modal_attention_blocks": tune.choice([1, 2, 4, 8]),
    "num_cross_modal_attention_ff_dim": tune.choice([64, 128, 256, 512, 728, 1024]),
    "batch_size": tune.choice([8, 16, 32, 64, 128, 256]),
    "lr": tune.loguniform(1e-5, 1e-2),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "alpha": tune.uniform(0.0, 1.0),
    "gamma": tune.uniform(0.0, 5.0),
}


def train_function(config, device: str, text_encoder, graph_encoder, output_classes, dataset: Dataset,
                   max_epochs: int = 10):
    model = MultiModalModelForClassification(
        text_encoder=text_encoder,
        graph_encoder=graph_encoder,
        self_attention_heads=config["self_attention_heads"],
        embedding_dim=config["embedding_dim"],
        num_cross_modal_attention_blocks=config["num_cross_modal_attention_blocks"],
        num_cross_modal_attention_heads=config["num_cross_modal_attention_heads"],
        self_attn_ff_dim=config["self_attn_ff_dim"],
        num_cross_modal_attention_ff_dim=config["num_cross_modal_attention_ff_dim"],
        output_channels=output_classes
    )
    move_to_device(model)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=astrorag_collate_fn,
    )
    criterion = FocalLoss(alpha=config["alpha"], gamma=config["gamma"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        loss_sum = 0.0
        loss_count = 0
        all_true = []
        all_probs = []
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch + 1}/{max_epochs}"):
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            graph_data = batch["graph_data"]
            graph_data.x = graph_data.x.to(device)
            graph_data.edge_index = graph_data.edge_index.to(device)
            graph_data.batch = graph_data.batch.to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            output = model(text_input_ids, text_attention_mask, graph_data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            loss_count += 1

            probs = torch.softmax(output, dim=1)[:, 1]
            all_probs.extend(probs.detach().cpu().tolist())
            all_true.extend(labels.detach().cpu().tolist())

        avg_loss = loss_sum / loss_count if loss_count > 0 else 0

        try:
            roc_auc = roc_auc_score(all_true, all_probs)
        except ValueError:
            roc_auc = float("nan")

        checkpoint_state = {
            "epoch": epoch + 1,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_state, fp)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            tune.report({"loss": avg_loss, "roc_auc": roc_auc}, checkpoint=checkpoint)


def load_data(dataset_root_dir: str, search_sample_size: int, text_encoder_model_id: str):
    full_dataset = AstroturfCampaignMultiModalDataset(
        json_dir=f"{dataset_root_dir}/train/graphs",
        model_id=text_encoder_model_id,
    )
    sample_size = min(search_sample_size, len(full_dataset))
    indices = random.sample(range(len(full_dataset)), sample_size)
    return Subset(full_dataset, indices)


@app.command()
def main(
        dataset_root_dir: str = typer.Option("./data", help="Path to the dataset root directory."),
        search_results_output_file_path: str = typer.Option("./", help="Path to save search results."),
        search_sample_size: int = typer.Option(1000, help="Number of samples to use for search."),
        text_encoder_model_id: str = typer.Option("answerdotai/ModernBERT-base",
                                                  help="Pretrained model ID for text encoder."),
        graph_encoder_model_path: str = typer.Option("path/to/graph_encoder.pth",
                                                     help="Path to the pre-trained graph encoder."),
        max_epochs: int = typer.Option(10, help="Maximum number of epochs for training."),
        output_classes: int = typer.Option(2, help="Number of output classes"),
):
    text_encoder = AutoModel.from_pretrained(text_encoder_model_id)
    graph_encoder = load_pre_trained_graph_encoder(
        model_path=graph_encoder_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    dataset = load_data(
        dataset_root_dir=dataset_root_dir,
        search_sample_size=search_sample_size,
        text_encoder_model_id=text_encoder_model_id
    )
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    trainable = tune.with_parameters(
        train_function,
        device="cuda" if torch.cuda.is_available() else "cpu",
        text_encoder=text_encoder,
        graph_encoder=graph_encoder,
        output_classes=output_classes,
        dataset=dataset,
        max_epochs=max_epochs,
    )
    abs_storage_path = Path(search_results_output_file_path).absolute().as_uri()
    analysis = tune.run(
        trainable,
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=search_space,
        num_samples=10,
        storage_path=abs_storage_path,
        name="multi_modal_search",
        scheduler=scheduler,
        stop={"training_iteration": max_epochs},
    )
    best_trial = analysis.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final ROC AUC: {best_trial.last_result['roc_auc']}")
    output_file = Path(search_results_output_file_path) / "analysis.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(analysis, f)


if __name__ == "__main__":
    app()
