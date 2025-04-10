from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
from transformers import AutoTokenizer
from torch_geometric.data import Batch


class DummyMultiModalDataset(Dataset):
    def __init__(self,
                 num_samples: int = 100,
                 text_length: int = 20,
                 text_model_name: str = "bert-base-uncased"):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            text_length (int): Maximum token length for the text input.
            text_model_name (str): Hugging Face model name used to initialize the tokenizer.
        """
        self.num_samples = num_samples
        self.text_length = text_length

        # Initialize the tokenizer from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # -----------
        # Graph Data
        # -----------
        # Create node features: 72 nodes each with 10 features, shape [72, 10]
        x = torch.randn(72, 10, dtype=torch.float)

        # Create edge_index: a tensor with shape [2, 71] representing 71 random edges
        edge_index = torch.randint(0, 72, (2, 71), dtype=torch.long)

        # Create a dummy label tensor (for binary classification) with shape [1]
        y = torch.randint(0, 2, (1,), dtype=torch.long)

        # Create a batch tensor: for a single graph, all nodes have the same batch index, here 0.
        batch = torch.zeros(72, dtype=torch.long)

        # Create a PyTorch Geometric Data object for the graph
        graph_data = Data(x=x, edge_index=edge_index, y=y, batch=batch)

        # -----------
        # Text Data
        # -----------
        # Create a dummy text sample (with index for variability)
        dummy_text = f"This is a dummy sentence number {idx} for testing multimodal input."

        # Tokenize using the specified tokenizer, padding/truncating to self.text_length
        tokenized = self.tokenizer(dummy_text,
                                   max_length=self.text_length,
                                   padding='max_length',
                                   truncation=True,
                                   return_tensors='pt')

        # Remove the extra batch dimension from the tokenized outputs; final shape is [text_length]
        text_input_ids = tokenized['input_ids'].squeeze(0)
        text_attention_mask = tokenized['attention_mask'].squeeze(0)

        # Return the data and label (convert label to a scalar using .item())
        return {
            'text_input_ids': text_input_ids,  # Shape: [text_length]
            'text_attention_mask': text_attention_mask,  # Shape: [text_length]
            'graph_data': graph_data  # Graph Data object with x, edge_index, y, batch
        }, y.item()  # Return the graph label as a scalar integer


def multimodal_collate_fn(batch):
    # Unzip batch elements into data dictionaries and corresponding labels
    data_dicts, labels = zip(*batch)

    text_input_ids = torch.stack([d['text_input_ids'] for d in data_dicts], dim=0)
    text_attention_mask = torch.stack([d['text_attention_mask'] for d in data_dicts], dim=0)

    # Create a batched graph using GeoBatch.from_data_list
    graph_data = Batch.from_data_list([d['graph_data'] for d in data_dicts])

    # Convert labels tuple (of ints) into a tensor.
    labels = torch.tensor(labels)

    return {
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'graph_data': graph_data,
        'labels': labels
    }


dataset = DummyMultiModalDataset(num_samples=10, text_length=20)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=multimodal_collate_fn)
