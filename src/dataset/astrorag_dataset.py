import os 
import json 

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class AstroRagDataset(Dataset):
    def __init__(self,json_dir,text_tokenizer, max_length=256,use_all_node_text=False):

        """
        Dataset for loading and processing the AstroRAg dataset.
        Args:
            :params json_dir (str): Dir->JSON files.
            :params text_tokenizer (Tokenizer): Tokenizer
            :params max_length (int): Maximum length for tokenized text.
            :params use_all_node_text (bool): If True, use all node text for each graph. (let me know if you want to change this)
        """

        self.json_dir = json_dir
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.use_all_node_text = use_all_node_text

        self.samples = []
        self._load_json_files()

    def _load_json_files(self):
        """
        Load all JSON files from Dir
        """
        if not os.path.exists(self.json_dir):
            raise ValueError(f"JSON directory {self.json_dir} does not exist.")
        
        for filename in os.listdir(self.json_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.json_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    label = 1 if data.get('label','').lower() == 'real' else 0

                    self.samples.append({
                        'file_path': file_path,
                        'label': label,
                    })
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)
    
    def _process_graph(self, data):
        """
        Process graph data from JSON
        
        Args:
           :params data (dict): Loaded JSON data
            
        Returns:
            Data: PyTorch Geometric Data object and text data
        """
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])

        # Creates Edge List
        # i.e node ID mapping for edges construction 
        node_id_map = {}

        # Extract node features and text
        node_features = []
        text_data = []

        # I'm gonna extract following features 
        # TODO: Can be modfied later (Optional)
        feature_names = ['followers_count', 'following_count', 'verified',
            'protected', 'favourites_count', 'listed_count', 'statuses_count', 'delay'
        ]
        for idx, node in enumerate(nodes):
            # First convert node ID to index
            node_id_map[node.get('id')] = idx
            # Extract features
            features = []
            for feature in feature_names:
                value = node.get(feature, 0)
                features.append(float(value)) # Convert to float for numerical stability
            
            node_features.append(features)
            text_data.append(node.get('tweet_text', ''))

        # Process edges to create edge_index
        edge_pairs = []
        for edge in edges:
            source_idx = node_id_map.get(edge.get('source')) 
            target_idx = node_id_map.get(edge.get('target'))
            
            if source_idx is not None and target_idx is not None:
                edge_pairs.append((source_idx, target_idx))

        # Convert node features to tensor
        if node_features:
            x = torch.tensor(node_features, dtype=torch.float)
        else:
            x = torch.zeros((0, len(feature_names)), dtype=torch.float)
        
        # Convert edge_index to tensor, ensuring proper format for PyG
        # Convert edge list to PyTorch tensor(edge_index)
        # Stored column-wise (i.e., first row contains source nodes, second row contains target nodes)
        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous() 
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Create a PyTorch Geometric Data object (graph)
        # Sample format: [num_nodes, num_features]
        # The graph object will contain the node features and edge indices
        # (See https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs)

        # Note: num_nodes is the number of nodes in the graph
        # Note: num_features is the number of features per node
        # Note: num_edges is the number of edges in the graph

      
        graph = Data(
            x=x, 
            edge_index=edge_index,
            num_nodes=len(nodes)
        )
        
        return graph, text_data
    

    def _get_text_input(self, text_data):
        """
        Process text data 
        
        Args:
           :params text_data (list): strings from nodes
            
        Returns:
            dict: Tokenized text for model input
        """
        if not text_data:
            # Fallback: for empty text (Returns zeros)
            # TODO: Let me know if you want to change this 
            return {
                'input_ids': torch.zeros(self.max_text_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_text_length, dtype=torch.long)
            }
        
        if self.use_all_node_text:
            # Combine all texts with separator
            # When we combine all node texts, we can use a separator to distinguish them (optional)
            combined_text = " ".join(text for text in text_data if text)
            text_to_process = combined_text
        else:
            # Use only the root node text (first node)
            # TODO: Again, let me know if you want to change this 
            text_to_process = text_data[0] if text_data else ""
        
        # Embeddings for text
        encoding = self.tokenizer(
            text_to_process,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0), # Shape: [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0) # Shape: [max_length]
        }


    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        Args:
            :params idx (int): Index to retrieve.

        Returns:
            tuple: (graph_data,label).

        """

        sample = self.samples[idx]
        json_path = sample['json_path']
        label = sample['label']
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        

        graph, text_data = self._process_graph(data) # Graph data
        text_inputs = self._get_text_input(text_data) # Text data
        
        # Return the graph_data dic and label
        return {
            'text_input_ids': text_inputs['input_ids'],
            'text_attention_mask': text_inputs['attention_mask'],
            'graph_data': graph
        }, label



    def astrorag_collate_fn(batch):
        """
        Collate function for the AstroRag dataset
        
        Args:
            :params batch: List of (data_dict, label)
        
        Returns:
            dict: Batch of data(text and graph) and labels
        """
        data_dicts, labels = zip(*batch)
        # Stack image tensors (pixel_values) into a batch.
        # Convert text input IDs and attention masks to tensors
        text_input_ids = torch.stack([d['text_input_ids'] for d in data_dicts])
        text_attention_mask = torch.stack([d['text_attention_mask'] for d in data_dicts])
        
        # Batch the graph data using PyG's Batch
        graphs = Batch.from_data_list([d['graph_data'] for d in data_dicts])
        
        # Convert labels to tensor 
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'graph_data': graphs,
            'labels': labels
        } 