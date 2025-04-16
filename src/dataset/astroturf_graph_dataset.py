import json
import os
import os.path as osp

import torch
from torch.serialization import safe_globals
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage


class AstroturfCampaignGraphDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        """
        Args:
            root (str): Root directory containing:
                        - train/graphs/*.json
                        - test/graphs/*.json
            split (str): Which split to load; either 'train' or 'test'.
            transform (callable, optional): A function/transform that takes in a Data object
                                            and returns a transformed version.
            pre_transform (callable, optional): A function/transform that takes in a Data object
                                                and returns a transformed version.
        """
        self.split = split  # 'train' or 'test'
        super(AstroturfCampaignGraphDataset, self).__init__(root, transform, pre_transform)
        # Allow safe globals for certain PyG types during unpickling.
        with safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        # JSON files now live under root/{split}/graphs/
        return osp.join(self.root, self.split, 'graphs')

    @property
    def raw_file_names(self):
        # Return all JSON files in the chosen split/graphs folder.
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.json')]

    @property
    def processed_dir(self):
        # Processed files will still go under root/processed
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        # Use split-specific naming.
        return [f"data_{self.split}.pt"]

    def download(self):
        # No-op: data assumed to already exist under raw_dir.
        pass

    def process(self):
        data_list = []
        for file_name in self.raw_file_names:
            file_path = osp.join(self.raw_dir, file_name)
            with open(file_path, 'r') as f:
                graph_json = json.load(f)

            # 1. Label
            label_str = graph_json.get("label", "real").lower()
            label_val = 0 if label_str == "real" else 1

            # 2. Nodes → feature vectors
            nodes = graph_json.get("nodes", [])
            features_list = []
            node_id_to_index = {}
            for idx, node in enumerate(nodes):
                node_id_to_index[node.get("id")] = idx
                feature_vector = [
                    int(node.get("verified", 0)),
                    node.get("followers_count", 0),
                    node.get("following_count", 0),
                    node.get("statuses_count", 0),
                    node.get("favourites_count", 0),
                    node.get("listed_count", 0),
                    len((node.get("associated_user_profile_description") or "").split()),
                    node.get("delay", 0),
                    len((node.get("tweet_text") or "").split())
                ]
                features_list.append(feature_vector)
            x = torch.tensor(features_list, dtype=torch.float)

            # 3. Edges
            edges = graph_json.get("edges", [])
            src_list, tgt_list = [], []
            for edge in edges:
                s, t = edge.get("source"), edge.get("target")
                if s in node_id_to_index and t in node_id_to_index:
                    src_list.append(node_id_to_index[s])
                    tgt_list.append(node_id_to_index[t])
            edge_index = (torch.tensor([src_list, tgt_list], dtype=torch.long)
                          if src_list else torch.empty((2, 0), dtype=torch.long))

            # 4. Label tensor
            y = torch.tensor([label_val], dtype=torch.long)

            # 5. Build Data and append
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        # Collate & save
        os.makedirs(self.processed_dir, exist_ok=True)
        data, slices = self.collate(data_list)
        torch.save((data, slices),
                   osp.join(self.processed_dir, self.processed_file_names[0]))

    @property
    def num_features(self):
        if len(self) > 0 and hasattr(self[0], "x"):
            return self[0].x.size(1)
        return 0

    @property
    def num_classes(self):
        if hasattr(self, "data") and hasattr(self.data, "y"):
            return int(self.data.y.max().item() + 1)
        if len(self) > 0 and hasattr(self[0], "y"):
            return int(self[0].y.max().item() + 1)
        return 0
