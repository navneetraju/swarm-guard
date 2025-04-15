import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, LayerNorm


class UPFDGraphSageNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.3):
        super(UPFDGraphSageNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.norm1 = LayerNorm(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)

        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.norm3 = LayerNorm(hidden_channels)

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        """
        The forward pass
        :param x: Graph node features
        :param edge_index: Graph edge indices
        :param batch: Batch indices
        :return: (classification_logits, node_embeddings, averaged_graph_level_embedding)
        """
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h1 = self.norm1(h1)

        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = self.norm2(h2 + h1)

        h3 = self.conv3(h2, edge_index)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=self.dropout, training=self.training)
        h3 = self.norm3(h2 + h3)

        averaged_embedding = global_mean_pool(h3, batch)
        out = self.classifier(averaged_embedding)
        return out, h3, averaged_embedding
