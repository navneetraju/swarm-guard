import argparse

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from src.dataset import AstroturfCampaignMultiModalDataset, astrorag_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained BERT classifier on test set")
    parser.add_argument("--text_encoder_model_id", type=str, default="answerdotai/ModernBERT-base",
                        help="Pretrained BERT classification model ID or path")
    parser.add_argument("--test_data_dir", type=str, required=True,
                        help="Directory containing test JSON files (e.g., dataset/test)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="Number of output classes")
    return parser.parse_args()


def main():
    args = parse_args()

    # Select device: MPS if available, else CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print("Using MPS device for inference")
    else:
        device = torch.device('cpu')
        print("Using CPU for inference")

    # Load pretrained BERT for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        args.text_encoder_model_id,
        num_labels=args.num_labels,
        return_dict=True
    )
    model.to(device)
    model.eval()

    # Prepare test data loader (text only; graph data ignored)
    test_dataset = AstroturfCampaignMultiModalDataset(
        json_dir=args.test_data_dir,
        model_id=args.text_encoder_model_id
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=astrorag_collate_fn
    )

    all_labels = []
    all_preds = []
    all_probs = []

    # Iterate over test set
    for batch in tqdm(test_loader, desc="Evaluating BERT classifier"):
        input_ids = batch['text_input_ids'].to(device)
        attention_mask = batch['text_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())

    # Handle empty dataset
    if not all_labels:
        print("No test samples found. Please check the test data directory.")
        return

    # Compute and print metrics
    print("Classification Report:\n", classification_report(all_labels, all_preds, digits=4))
    print(f"ROC AUC Score: {roc_auc_score(all_labels, all_probs):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))


if __name__ == '__main__':
    main()
