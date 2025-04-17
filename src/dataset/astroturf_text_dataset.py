import json
import os

import torch
from transformers import AutoTokenizer


class AstroturfTextDataset(torch.utils.data.Dataset):
    def __init__(self, json_dir: str, model_id: str, max_length: int = 280):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_length = max_length
        self.samples = []

        for fname in os.listdir(json_dir):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(json_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            label = 0 if data.get('label', 'real').lower() == 'real' else 1
            text = data.get('nodes', [{}])[0].get('tweet_text', '') or ''
            if text:
                self.samples.append((text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def labels(self):
        return [lab for _, lab in self.samples]
