import torch.nn as nn
import torch.nn.functional as F


class CrossModelAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, feed_forward_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mha_text_graph = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.mha_graph_text = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

        self.ff_text_graph_1 = nn.Linear(in_features=embedding_dim, out_features=feed_forward_dim)
        self.ff_graph_text_1 = nn.Linear(in_features=embedding_dim, out_features=feed_forward_dim)

        self.ff_text_graph_2 = nn.Linear(in_features=feed_forward_dim, out_features=embedding_dim)
        self.ff_graph_text_2 = nn.Linear(in_features=feed_forward_dim, out_features=embedding_dim)

        self.text_graph_norm = nn.LayerNorm(embedding_dim)
        self.graph_text_norm = nn.LayerNorm(embedding_dim)

    def forward(self, text_embedding, graph_embedding):
        mha_text_graph_out, _ = self.mha_text_graph(text_embedding, graph_embedding, graph_embedding)
        mha_graph_text_out, _ = self.mha_graph_text(graph_embedding, text_embedding, text_embedding)

        text_graph_out = F.relu(self.ff_text_graph_1(mha_text_graph_out))
        graph_text_out = F.relu(self.ff_graph_text_1(mha_graph_text_out))

        text_graph_out = self.ff_text_graph_2(text_graph_out)
        graph_text_out = self.ff_graph_text_2(graph_text_out)

        text_out = self.text_graph_norm(text_graph_out + mha_text_graph_out)
        graph_out = self.graph_text_norm(graph_text_out + mha_graph_text_out)

        return text_out, graph_out
