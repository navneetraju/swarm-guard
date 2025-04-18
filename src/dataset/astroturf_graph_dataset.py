import json
import os
import os.path as osp

import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data


class AstroturfCampaignGraphDataset(IterableDataset):
    """
    An iterable PyG dataset that streams graphs from raw JSON files without loading
    everything into memory.

    Args:
        root (str): Root directory containing `train/graphs/` and `test/graphs/`.
        split (str): Which split to iterate; either 'train' or 'test'.
        transform (callable, optional): A function/transform that takes in a Data object
                                        and returns a transformed version.
        shuffle (bool, optional): Whether to shuffle file order each epoch.
    """

    def __init__(self, root: str, split: str = 'train', transform=None, shuffle: bool = False):
        self.root = root
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.raw_dir = osp.join(self.root, self.split, 'graphs')
        self._file_list = [f for f in os.listdir(self.raw_dir) if f.endswith('.json')]

    def __iter__(self):
        files = list(self._file_list)
        if self.shuffle:
            random.shuffle(files)
        for file_name in files:
            path = osp.join(self.raw_dir, file_name)
            with open(path, 'r') as f:
                graph_json = json.load(f)

            # 1. Label
            label = graph_json.get("label", "real").lower()
            y = torch.tensor([0 if label == "real" else 1], dtype=torch.long)

            # 2. Node features
            nodes = graph_json.get("nodes", [])
            node_id_to_idx = {}
            feats = []
            for idx, node in enumerate(nodes):
                node_id_to_idx[node.get("id")] = idx
                feats.append([
                    int(node.get("verified", 0)),
                    node.get("followers_count", 0),
                    node.get("following_count", 0),
                    node.get("statuses_count", 0),
                    node.get("favourites_count", 0),
                    node.get("listed_count", 0),
                    len((node.get("associated_user_profile_description") or "").split()),
                    node.get("delay", 0),
                    len((node.get("tweet_text") or "").split())
                ])
            x = torch.tensor(feats, dtype=torch.float)

            # 3. Edge index
            edges = graph_json.get("edges", [])
            src, dst = [], []
            for edge in edges:
                s, t = edge.get("source"), edge.get("target")
                if s in node_id_to_idx and t in node_id_to_idx:
                    src.append(node_id_to_idx[s])
                    dst.append(node_id_to_idx[t])
            edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            if self.transform:
                data = self.transform(data)
            yield data
