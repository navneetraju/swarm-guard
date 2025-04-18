import json
import os
import os.path as osp
import random

import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data


class AstroturfCampaignGraphDataset(IterableDataset):
    def __init__(self, root: str, split: str = "train", transform=None, shuffle=False):
        self.root = osp.abspath(osp.expanduser(root))
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.raw_dir = osp.join(self.root, self.split, "graphs")
        if not osp.isdir(self.raw_dir):
            raise FileNotFoundError(f"Raw directory not found: {self.raw_dir}")
        self._file_list = [f for f in os.listdir(self.raw_dir) if f.endswith(".json")]

    def __len__(self):
        return len(self._file_list)

    def __iter__(self):
        files = list(self._file_list)
        if self.shuffle:
            random.shuffle(files)
        for fname in files:
            with open(osp.join(self.raw_dir, fname), "r") as f:
                g = json.load(f)

            label = 0 if g.get("label", "real").lower() == "real" else 1
            y = torch.tensor([label], dtype=torch.long)

            id2idx, feats = {}, []
            for i, node in enumerate(g.get("nodes", [])):
                id2idx[node["id"]] = i
                feats.append(
                    [
                        int(node.get("verified", 0)),
                        node.get("followers_count", 0),
                        node.get("following_count", 0),
                        node.get("statuses_count", 0),
                        node.get("favourites_count", 0),
                        node.get("listed_count", 0),
                        len((node.get("associated_user_profile_description") or "").split()),
                        node.get("delay", 0),
                        len((node.get("tweet_text") or "").split()),
                    ]
                )
            x = torch.tensor(feats, dtype=torch.float)

            src, dst = [], []
            for e in g.get("edges", []):
                s, t = e["source"], e["target"]
                if s in id2idx and t in id2idx:
                    src.append(id2idx[s])
                    dst.append(id2idx[t])
            edge_index = (
                torch.tensor([src, dst], dtype=torch.long)
                if src
                else torch.empty((2, 0), dtype=torch.long)
            )

            data = Data(x=x, edge_index=edge_index, y=y)
            if self.transform:
                data = self.transform(data)
            yield data
