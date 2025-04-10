import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm


class UPFDGraphSageNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.3):
        super(UPFDGraphSageNet, self).__init__()
        self.dropout = dropout

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        """
        The forward pass
        :param x: Graph node features
        :param edge_index: Graph edge indices
        :param batch: Batch indices
        :return: (classification_logits, node_embeddings, averaged_graph_level_embedding)
        """
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        averaged_embedding = global_mean_pool(h, batch)
        out = self.classifier(averaged_embedding)
        return out, h, averaged_embedding
