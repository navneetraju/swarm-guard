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
            root (str): Root directory containing subfolders 'train' and 'test'.
            split (str): Which split to load; must be either 'train' or 'test'.
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
        # Data is stored in a subfolder named after the split.
        return osp.join(self.root, self.split)

    @property
    def raw_file_names(self):
        # Return all JSON files in the chosen split folder.
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.json')]

    @property
    def processed_dir(self):
        # Processed files will be stored in the 'processed' folder under the root.
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        # Use split-specific naming.
        return [f"data_{self.split}.pt"]

    def download(self):
        # This method is left empty because data files are assumed to already exist.
        pass

    def process(self):
        data_list = []
        # Process each JSON graph file in the raw directory.
        for file_name in self.raw_file_names:
            file_path = osp.join(self.raw_dir, file_name)
            with open(file_path, 'r') as f:
                graph_json = json.load(f)

            # 1. Process graph-level label.
            # UPFD convention: label "real" -> 0, "fake" -> 1.
            label_str = graph_json.get("label", "real").lower()
            label_val = 0 if label_str == "real" else 1

            # 2. Process nodes.
            nodes = graph_json.get("nodes", [])
            features_list = []
            node_id_to_index = {}
            for idx, node in enumerate(nodes):
                node_id = node.get("id")
                node_id_to_index[node_id] = idx

                # Transformation steps (mimicking UPFD transformation for available fields):
                verified = int(node.get("verified", 0))
                followers_count = node.get("followers_count", 0)
                following_count = node.get("following_count", 0)  # used as friends count
                statuses_count = node.get("statuses_count", 0)
                favourites_count = node.get("favourites_count", 0)
                listed_count = node.get("listed_count", 0)
                description = node.get("associated_user_profile_description", "") or ""
                description_word_count = len(description.split())
                tweet_text = node.get("tweet_text", "") or ""
                tweet_word_count = len(tweet_text.split())
                delay = node.get("delay", 0)

                # Build the feature vector:
                # [Verified, Followers, Following, Statuses, Favourites, Listed,
                #  Description Word Count, Delay, Tweet Word Count]
                feature_vector = [
                    verified,
                    followers_count,
                    following_count,
                    statuses_count,
                    favourites_count,
                    listed_count,
                    description_word_count,
                    delay,
                    tweet_word_count
                ]
                features_list.append(feature_vector)

            # Create a tensor from the list of feature vectors.
            x = torch.tensor(features_list, dtype=torch.float)

            # 3. Process edges.
            edges = graph_json.get("edges", [])
            src_list = []
            tgt_list = []
            for edge in edges:
                src_id = edge.get("source")
                tgt_id = edge.get("target")
                # Look up the indices based on node 'id'
                if src_id in node_id_to_index and tgt_id in node_id_to_index:
                    src_list.append(node_id_to_index[src_id])
                    tgt_list.append(node_id_to_index[tgt_id])
            if len(src_list) > 0:
                edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # 4. Create the graph label tensor.
            y = torch.tensor([label_val], dtype=torch.long)

            # 5. Create the Data object for the graph.
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        # Collate the list of Data objects and save.
        os.makedirs(self.processed_dir, exist_ok=True)
        data, slices = self.collate(data_list)
        processed_file = osp.join(self.processed_dir, self.processed_file_names[0])
        torch.save((data, slices), processed_file)

    @property
    def num_features(self):
        """
        Returns the number of features per node.
        It is determined from the first graph's node feature matrix.
        """
        if len(self) > 0 and hasattr(self[0], "x") and self[0].x is not None:
            return self[0].x.size(1)
        return 0

    @property
    def num_classes(self):
        """
        Returns the number of classes (i.e. maximum label value + 1).
        Uses the collated data if available; otherwise, falls back to the first graph.
        """
        if hasattr(self, "data") and hasattr(self.data, "y") and self.data.y is not None:
            return int(self.data.y.max().item() + 1)
        if len(self) > 0 and hasattr(self[0], "y") and self[0].y is not None:
            return int(self[0].y.max().item() + 1)
        return 0
