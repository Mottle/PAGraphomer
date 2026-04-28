import torch
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head
from torch_geometric.nn import global_add_pool, global_max_pool


@register_head("san_graph")
class SANGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.
    Readout: sum_pool + max_pool concatenated → MLP classifier.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        input_dim = 2 * dim_in
        list_FC_layers = [
            nn.Linear(input_dim // 2**l, input_dim // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_FC_layers.append(nn.Linear(input_dim // 2**L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]()

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = torch.cat(
            [
                global_add_pool(batch.x, batch.batch),
                global_max_pool(batch.x, batch.batch),
            ],
            dim=-1,
        )
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
