import os
import random
from typing import Dict, Iterator, Any, Optional

import torch
from torch.utils.data import IterableDataset
from PIL import Image
from transformers import AutoImageProcessor


class AstroturfImageDataset(IterableDataset):
    def __init__(self,root: str,model_id: str,split: str = "train",transform=None,shuffle: bool = False,cache_processor: bool = True,):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.model_id = model_id
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.image_dir = os.path.join(self.root, self.split, "images")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self._file_list = [f for f in os.listdir(self.image_dir) if f.lower().endswith(".jpg")]
        if self.shuffle:
            random.shuffle(self._file_list)

        self._processor = AutoImageProcessor.from_pretrained(model_id) if cache_processor else None

    def __len__(self) -> int:
        return len(self._file_list)

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        return self._processor

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        file_list = list(self._file_list)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = int(len(file_list) / worker_info.num_workers)
            start = worker_info.id * per_worker
            end = start + per_worker
            file_list = file_list[start:end]

        for fname in file_list:
            image_path = os.path.join(self.image_dir, fname)
            try:
                image = Image.open(image_path).convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()} 

            yield {
                **inputs,
                "image_path": image_path
            }
