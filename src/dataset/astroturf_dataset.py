import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer, AutoImageProcessor


class AstroturfCampaignMultiModalDataset(Dataset):
    def __init__(self, json_dir, image_dir, text_model_id, vision_model_id, max_length=280, use_all_node_text=False):
        """
        Dataset for loading and processing the Astroturf Campaigns dataset.

        Args:
            json_dir (str): Directory containing JSON files.
            model_id (str): Pretrained model identifier for tokenization.
            max_length (int): Maximum length for tokenized text.
            use_all_node_text (bool): If True, use all node text for each graph.
        """
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
        self.vision_processor = AutoImageProcessor.from_pretrained(vision_model_id)
        self.max_length = max_length
        self.use_all_node_text = use_all_node_text
        self.image_filenames = {}
        self.samples = []
        self._read_image_directory()
        self._load_json_files()

    def _read_image_directory(self):
        """
        Read image filenames from the directory and store them in a dictionary.
        """
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory {self.image_dir} does not exist.")

        for filename in os.listdir(self.image_dir):
            # example path: swarm_guard_dataset/train/images/1234567890_1234567890.jpg
            if filename.endswith('.jpg'):
                # Extract the image ID from the filename
                image_id = filename.split('.')[0]
                # Store the image ID and its corresponding filename
                self.image_filenames[image_id] = os.path.join(self.image_dir, filename)

    def _load_json_files(self):
        """
        Load all JSON files from the directory.
        """
        if not os.path.exists(self.json_dir):
            raise ValueError(f"JSON directory {self.json_dir} does not exist.")

        for filename in os.listdir(self.json_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.json_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Map "real" -> 0, "fake" -> 1 (same as the graph-only dataset)
                    label = 0 if data.get('label', 'real').lower() == 'real' else 1
                    original_post = data['nodes'][0]
                    image_file_name = f"{original_post.get('id')}_{original_post.get('user_id')}"
                    if image_file_name not in self.image_filenames:
                        continue
                    self.samples.append({
                        'file_path': file_path,
                        'image_path': self.image_filenames[image_file_name],
                        'label': label,
                    })

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)

    def _process_graph(self, data, label):
        """
        Process graph data from JSON.

        Args:
            data (dict): Loaded JSON data.

        Returns:
            graph (Data): PyTorch Geometric Data object.
            text_data (list): List of text strings from nodes.
        """
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])

        node_id_map = {}
        node_features = []
        text_data = []

        for idx, node in enumerate(nodes):
            node_id = node.get('id')
            node_id_map[node_id] = idx

            # Extract features following the graph-only dataset transformation:
            verified = int(node.get('verified', 0))
            followers_count = node.get('followers_count', 0)
            following_count = node.get('following_count', 0)
            statuses_count = node.get('statuses_count', 0)
            favourites_count = node.get('favourites_count', 0)
            listed_count = node.get('listed_count', 0)
            description = node.get('associated_user_profile_description', "") or ""
            description_word_count = len(description.split())
            tweet_text = node.get('tweet_text', "") or ""
            tweet_word_count = len(tweet_text.split())
            delay = node.get('delay', 0)

            features = [
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
            node_features.append(features)
            text_data.append(tweet_text)

        # Process edges to create edge_index tensor
        edge_pairs = []
        for edge in edges:
            source_idx = node_id_map.get(edge.get('source'))
            target_idx = node_id_map.get(edge.get('target'))
            if source_idx is not None and target_idx is not None:
                edge_pairs.append((source_idx, target_idx))

        # Create a tensor of node features
        x = torch.tensor(node_features, dtype=torch.float) if node_features else torch.zeros((0, 9), dtype=torch.float)

        # Create edge_index tensor
        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Create the PyTorch Geometric Data object.
        graph = Data(x=x, edge_index=edge_index, num_nodes=len(nodes),
                     y=torch.tensor([label], dtype=torch.long))
        return graph, text_data

    def _get_text_input(self, text_data):
        """
        Process text data using the tokenizer.

        Args:
            text_data (list): List of text strings from nodes.

        Returns:
            dict: Tokenized text inputs containing input_ids and attention_mask.
        """
        if not text_data:
            raise ValueError("No text data available for tokenization for this sample.")

        if self.use_all_node_text:
            # Combine all node texts if required
            combined_text = " ".join(text for text in text_data if text)
            text_to_process = combined_text
        else:
            # Use only the text from the first node
            text_to_process = text_data[0] if text_data else ""

        encoding = self.text_tokenizer(
            text_to_process,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Shape: [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0)  # Shape: [max_length]
        }

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index to retrieve.

        Returns:
            tuple: (data_dict, label) where data_dict contains graph data and text inputs.
        """
        sample = self.samples[idx]
        json_path = sample['file_path']
        label = sample['label']
        image_path = sample['image_path']
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph, text_data = self._process_graph(data, label)
        text_inputs = self._get_text_input(text_data)
        image_processed = self._get_image_processed(image_path)
        return {
            'text_input_ids': text_inputs['input_ids'],
            'text_attention_mask': text_inputs['attention_mask'],
            'pixel_values': image_processed['pixel_values'].squeeze(0),
            'graph_data': graph,
        }, label

    def _get_image_processed(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_processed = self.vision_processor(images=image, return_tensors="pt")
        return image_processed


def astrorag_collate_fn(batch):
    """
    Collate function for the AstroRag dataset.

    Args:
        batch (list): List of (data_dict, label) tuples.

    Returns:
        dict: A dictionary containing batched text inputs, graph data, and labels.
    """
    data_dicts, labels = zip(*batch)
    text_input_ids = torch.stack([d['text_input_ids'] for d in data_dicts])
    text_attention_mask = torch.stack([d['text_attention_mask'] for d in data_dicts])
    graphs = Batch.from_data_list([d['graph_data'] for d in data_dicts])
    pixel_values = torch.stack([d['pixel_values'] for d in data_dicts])
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'graph_data': graphs,
        'pixel_values': pixel_values,
        'labels': labels
    }
