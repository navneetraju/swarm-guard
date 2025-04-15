import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from src.modules.multi_modal.cross_modal_attention import CrossModelAttentionBlock


class MultiModalModelForClassification(nn.Module):
    def __init__(self,
                 text_encoder: nn.Module,
                 graph_encoder: nn.Module,
                 self_attention_heads: int,
                 embedding_dim: int,
                 num_cross_modal_attention_blocks: int,
                 num_cross_modal_attention_heads: int,
                 self_attn_ff_dim: int,
                 num_cross_modal_attention_ff_dim: int,
                 output_channels: int):
        super().__init__()

        # Use the provided encoders and freeze them for PEFT.
        self.text_encoder = text_encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.graph_encoder = graph_encoder
        for param in self.graph_encoder.parameters():
            param.requires_grad = False

        # Assuming the text encoder has a config with hidden_size.
        self.text_embedding_size = self.text_encoder.config.hidden_size
        self.embedding_dim = embedding_dim

        ############ PROJECTION ############
        self.text_projection = nn.Linear(in_features=self.text_embedding_size, out_features=embedding_dim)
        # Adjust the in_features for the graph projection if needed.
        self.graph_projection = nn.Linear(in_features=graph_encoder.hidden_channels, out_features=embedding_dim)

        ############ SELF ATTENTION ############
        self.text_self_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                         num_heads=self_attention_heads,
                                                         batch_first=True)
        self.graph_self_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                          num_heads=self_attention_heads,
                                                          batch_first=True)
        self.text_self_attention_norm = nn.LayerNorm(embedding_dim)
        self.graph_self_attention_norm = nn.LayerNorm(embedding_dim)
        self.text_self_attention_ff1 = nn.Linear(in_features=embedding_dim, out_features=self_attn_ff_dim)
        self.text_self_attention_ff2 = nn.Linear(in_features=self_attn_ff_dim, out_features=embedding_dim)

        self.graph_self_attention_ff1 = nn.Linear(in_features=embedding_dim, out_features=self_attn_ff_dim)
        self.graph_self_attention_ff2 = nn.Linear(in_features=self_attn_ff_dim, out_features=embedding_dim)

        self.text_self_attention_ff_norm = nn.LayerNorm(embedding_dim)
        self.graph_self_attention_ff_norm = nn.LayerNorm(embedding_dim)

        ############ CROSS MODAL ATTENTION ############
        self.cross_modal_attention_blocks = nn.ModuleList([
            CrossModelAttentionBlock(embedding_dim=embedding_dim,
                                     num_heads=num_cross_modal_attention_heads,
                                     feed_forward_dim=num_cross_modal_attention_ff_dim)
            for _ in range(num_cross_modal_attention_blocks)
        ])

        ############ OUTPUT LAYER ############
        self.output_pre_norm = nn.LayerNorm(embedding_dim * 2)
        self.output_ff = nn.Linear(embedding_dim * 2, output_channels)

    def forward(self, text_input_ids, text_attention_mask, graph_data):
        text_embedding = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)[0]
        _, node_embeddings, _ = self.graph_encoder(graph_data.x, graph_data.edge_index, graph_data.batch)
        dense_graph_embeddings, mask = to_dense_batch(node_embeddings, graph_data.batch)

        ############ PROJECTION ############
        projected_text_embedding = self.text_projection(text_embedding)
        projected_graph_embedding = self.graph_projection(dense_graph_embeddings)

        ############ SELF ATTENTION ############
        text_self_attn_out, _ = self.text_self_attention(projected_text_embedding,
                                                         projected_text_embedding,
                                                         projected_text_embedding)
        graph_self_attn_out, _ = self.graph_self_attention(projected_graph_embedding,
                                                           projected_graph_embedding,
                                                           projected_graph_embedding,
                                                           key_padding_mask=~mask)
        text_self_attn_out = self.text_self_attention_norm(text_self_attn_out + projected_text_embedding)
        graph_self_attn_out = self.graph_self_attention_norm(graph_self_attn_out + projected_graph_embedding)

        text_ff_out = F.relu(self.text_self_attention_ff1(text_self_attn_out))
        graph_ff_out = F.relu(self.graph_self_attention_ff1(graph_self_attn_out))
        text_ff_out = self.text_self_attention_ff2(text_ff_out)
        graph_ff_out = self.graph_self_attention_ff2(graph_ff_out)
        text_ff_out = self.text_self_attention_ff_norm(text_self_attn_out + text_ff_out)
        graph_ff_out = self.graph_self_attention_ff_norm(graph_self_attn_out + graph_ff_out)

        ############ CROSS MODAL ATTENTION ############
        projected_text_embedding, projected_graph_embedding = text_ff_out, graph_ff_out
        for block in self.cross_modal_attention_blocks:
            projected_text_embedding, projected_graph_embedding = block(projected_text_embedding,
                                                                        projected_graph_embedding)

        ############ OUTPUT LAYER ############
        global_text_embedding = torch.mean(projected_text_embedding, dim=1)
        global_graph_embedding = torch.mean(projected_graph_embedding, dim=1)
        combined_embedding = torch.cat((global_text_embedding, global_graph_embedding), dim=-1)
        combined_embedding = self.output_pre_norm(combined_embedding)
        output = self.output_ff(combined_embedding)
        return output
