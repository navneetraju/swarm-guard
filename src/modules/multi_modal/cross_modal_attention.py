import torch.nn as nn
import torch.nn.functional as F


class CrossModelAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, feed_forward_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.ff_1 = nn.Linear(in_features=embedding_dim, out_features=feed_forward_dim)
        self.ff_2 = nn.Linear(in_features=feed_forward_dim, out_features=embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, embedding_a, embedding_b):
        mha_out, _ = self.mha(embedding_a, embedding_b, embedding_b)
        out = F.relu(self.ff_1(mha_out))
        out = self.ff_2(out)
        final_out = self.norm(out + mha_out)
        return final_out
