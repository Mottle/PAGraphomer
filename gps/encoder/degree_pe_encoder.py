import torch
import torch.nn as nn
from torch_geometric.utils import degree
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder("DegreePE")
class DegreePENodeEncoder(nn.Module):
    """Node degree positional encoding.

    Encodes scalar node degrees as positional features.
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        self.expand_x = expand_x

    def forward(self, batch):
        degrees = degree(batch.edge_index[0], num_nodes=batch.num_nodes)
        batch.deg = degrees.float().view(-1, 1)
        if self.expand_x:
            batch.x = torch.cat([batch.x, batch.deg], dim=-1)
        return batch
