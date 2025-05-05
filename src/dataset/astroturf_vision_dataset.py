import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor


def create_labeled_image_df(image_dir: str, json_dir: str) -> pd.DataFrame:
    """
    Here we match jpg images with tweet+user IDs from JSONs and assign labels.

    Args:
        image_dir (str): Path to the directory containing images.
        json_dir (str): Path to the directory containing JSON files.

    Returns:
        pd.DataFrame: DataFrame with columns: image_key, image_path, label, json_file
    """
    image_dict = {}  # key: tweetid_userid, value: full image path

    for fname in os.listdir(image_dir):
        if fname.endswith(".jpg"):
            key = fname.split(".")[0]
            image_dict[key] = os.path.join(image_dir, fname)

    rows = []

    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue

        with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
            data = json.load(f)

        label_str = data.get("label", "real").lower()
        label = 0 if label_str == "real" else 1

        for node in data.get("nodes", []):
            tweet_id = node.get("id")
            user_id = node.get("user_id")
            if not tweet_id or not user_id:
                continue

            key = f"{tweet_id}_{user_id}"
            if key in image_dict:
                rows.append({
                    "image_key": key,
                    "image_path": image_dict[key],
                    "label": label,
                    "json_file": json_file
                })

    return pd.DataFrame(rows)


class AstroImageDataset(Dataset):
    """
        df (pd.DataFrame): DataFrame with image_path and label columns.
        vision_model_id (str): Hugging Face model ID for the ViT processor.
    """

    def __init__(self, df: pd.DataFrame, vision_model_id: str):
        self.df = df.reset_index(drop=True)
        self.vision_processor = AutoImageProcessor.from_pretrained(vision_model_id)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = row["label"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image at {image_path}: {e}")

        processed = self.vision_processor(image, return_tensors="pt")

        return {
            # squeeze - removes the batch dimension → Resulting shape becomes[3, 224, 224], which is what PyTorch models expect for individual images.
            "pixel_values": processed["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
