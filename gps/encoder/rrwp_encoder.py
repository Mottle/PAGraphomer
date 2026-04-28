import warnings

import torch
from torch import nn
import torch_sparse
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter


def full_edge_index(edge_index, batch=None):
    """Return full batched edge index including non-edges and self loops."""
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="add")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    full_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        adj = torch.ones((n, n), dtype=torch.short, device=edge_index.device)
        full_index_list.append(
            adj.nonzero(as_tuple=False).t().contiguous() + cum_nodes[i]
        )

    return torch.cat(full_index_list, dim=1).contiguous()


@register_node_encoder("rrwp_linear")
class RRWPLinearNodeEncoder(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        out_dim,
        use_bias=False,
        batchnorm=False,
        layernorm=False,
        pe_name="rrwp",
    ):
        super().__init__()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.name = pe_name

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp = self.fc(batch[self.name])
        if self.batchnorm:
            rrwp = self.bn(rrwp)
        if self.layernorm:
            rrwp = self.ln(rrwp)

        if hasattr(batch, "x") and batch.x is not None:
            batch.x = batch.x + rrwp
        else:
            batch.x = rrwp
        return batch


@register_edge_encoder("rrwp_linear")
class RRWPLinearEdgeEncoder(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        out_dim,
        batchnorm=False,
        layernorm=False,
        use_bias=False,
        pad_to_full_graph=True,
        fill_value=0.0,
        add_node_attr_as_self_loop=False,
        overwrite_old_attr=False,
    ):
        super().__init__()
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr = overwrite_old_attr
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn(
                "batchnorm/layernorm may weaken RRWP shortest-path information"
            )

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = fill_value
        self.register_buffer(
            "padding", torch.ones(1, out_dim, dtype=torch.float) * fill_value
        )

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp_idx = batch.rrwp_index
        rrwp_val = self.fc(batch.rrwp_val)
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        if edge_attr is None:
            edge_attr = rrwp_val.new_zeros(edge_index.size(1), rrwp_val.size(1))

        if self.overwrite_old_attr:
            out_idx, out_val = rrwp_idx, rrwp_val
        else:
            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr,
                num_nodes=batch.num_nodes,
                fill_value=0.0,
            )
            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, rrwp_idx], dim=1),
                torch.cat([edge_attr, rrwp_val], dim=0),
                batch.num_nodes,
                batch.num_nodes,
                op="add",
            )

        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
                out_idx, out_val, batch.num_nodes, batch.num_nodes, op="add"
            )

        if self.batchnorm:
            out_val = self.bn(out_val)
        if self.layernorm:
            out_val = self.ln(out_val)

        batch.edge_index = out_idx
        batch.edge_attr = out_val
        return batch


@register_edge_encoder("masked_rrwp_linear")
class RRWPLinearEdgeMaskedEncoder(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        out_dim,
        batchnorm=False,
        layernorm=False,
        use_bias=False,
        fill_value=0.0,
        add_node_attr_as_self_loop=False,
        overwrite_old_attr=False,
        mask_index_name="edge_index",
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr = overwrite_old_attr
        self.mask_index_name = mask_index_name
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn(
                "batchnorm/layernorm may weaken RRWP shortest-path information"
            )

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fill_value = fill_value
        self.register_buffer(
            "padding", torch.ones(1, out_dim, dtype=torch.float) * fill_value
        )

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp_idx = batch.rrwp_index
        rrwp_val = self.fc(batch.rrwp_val)
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        mask_index = batch.get(self.mask_index_name, None)
        num_nodes = batch.num_nodes

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), rrwp_val.size(1))

        if self.overwrite_old_attr:
            out_idx, out_val = rrwp_idx, rrwp_val
        else:
            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, rrwp_idx], dim=1),
                torch.cat([edge_attr, rrwp_val], dim=0),
                batch.num_nodes,
                batch.num_nodes,
                op="add",
            )

        if mask_index is not None:
            mask_index, _ = add_remaining_self_loops(
                mask_index, None, num_nodes=batch.num_nodes
            )
            mask_val = mask_index.new_full((mask_index.size(1),), 1)
            mask_comp = mask_index.new_full((out_idx.size(1),), 0)
            mask_pad = mask_index.new_full((mask_index.size(1), out_val.size(1)), 0)
            _, masking = torch_sparse.coalesce(
                torch.cat([mask_index, out_idx], dim=1),
                torch.cat([mask_val, mask_comp], dim=0),
                m=num_nodes,
                n=num_nodes,
                op="max",
            )
            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([mask_index, out_idx], dim=1),
                torch.cat([mask_pad, out_val], dim=0),
                batch.num_nodes,
                batch.num_nodes,
                op="add",
            )
            masking = masking.type(torch.bool)
            out_idx, out_val = out_idx[:, masking], out_val[masking]

        if self.batchnorm:
            out_val = self.bn(out_val)
        if self.layernorm:
            out_val = self.ln(out_val)

        batch.edge_index = out_idx
        batch.edge_attr = out_val
        return batch


@register_edge_encoder("pad_to_full_graph")
class PadToFullGraphEdgeEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.pad_to_full_graph = True

    def forward(self, batch):
        out_idx, out_val = batch.edge_index, batch.edge_attr
        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = out_val.new_zeros(edge_index_full.size(1), out_val.size(1))
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
                out_idx, out_val, batch.num_nodes, batch.num_nodes, op="add"
            )

        batch.edge_index = out_idx
        batch.edge_attr = out_val
        return batch


class RRWPPairEncoder(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        out_dim,
        use_bias=False,
        edge_dim=None,
        rrwp_name="rrwp",
        pad_to_full_graph=True,
        fill_value=0.0,
        inject_edge_attr=True,
        add_adj_indicator=True,
    ):
        super().__init__()
        self.rrwp_name = rrwp_name
        self.pad_to_full_graph = pad_to_full_graph
        self.inject_edge_attr = inject_edge_attr
        self.add_adj_indicator = add_adj_indicator
        self.register_buffer(
            "padding", torch.ones(1, out_dim, dtype=torch.float) * fill_value
        )

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            torch.nn.init.zeros_(self.fc.bias)

        if edge_dim is None or not inject_edge_attr:
            self.edge_proj = None
        elif edge_dim == out_dim:
            self.edge_proj = nn.Identity()
        else:
            self.edge_proj = nn.Linear(edge_dim, out_dim, bias=use_bias)
            torch.nn.init.xavier_uniform_(self.edge_proj.weight)
            if self.edge_proj.bias is not None:
                torch.nn.init.zeros_(self.edge_proj.bias)

        if self.add_adj_indicator:
            self.edge_flag = nn.Parameter(torch.zeros(1, out_dim))
            nn.init.xavier_uniform_(self.edge_flag)
        else:
            self.edge_flag = None

    def forward(self, batch):
        rrwp_idx = getattr(batch, f"{self.rrwp_name}_index")
        rrwp_val = getattr(batch, f"{self.rrwp_name}_val")
        rrwp_val = self.fc(rrwp_val)

        pair_idx = rrwp_idx
        pair_val = rrwp_val

        if self.pad_to_full_graph:
            pair_full = getattr(batch, "srf_pair_index", None)
            if pair_full is None:
                pair_full = full_edge_index(pair_idx, batch=batch.batch)
            pair_pad = self.padding.repeat(pair_full.size(1), 1)
            pair_idx = torch.cat([pair_idx, pair_full], dim=1)
            pair_val = torch.cat([pair_val, pair_pad], dim=0)
            pair_idx, pair_val = torch_sparse.coalesce(
                pair_idx, pair_val, batch.num_nodes, batch.num_nodes, op="add"
            )

        if self.inject_edge_attr and getattr(batch, "edge_attr", None) is not None:
            edge_attr = batch.edge_attr
            edge_feat = (
                self.edge_proj(edge_attr) if self.edge_proj is not None else edge_attr
            )
            pair_idx = torch.cat([pair_idx, batch.edge_index], dim=1)
            pair_val = torch.cat([pair_val, edge_feat], dim=0)
            pair_idx, pair_val = torch_sparse.coalesce(
                pair_idx, pair_val, batch.num_nodes, batch.num_nodes, op="add"
            )

        if self.add_adj_indicator and self.edge_flag is not None:
            edge_flag = self.edge_flag.expand(batch.edge_index.size(1), -1)
            pair_idx = torch.cat([pair_idx, batch.edge_index], dim=1)
            pair_val = torch.cat([pair_val, edge_flag], dim=0)
            pair_idx, pair_val = torch_sparse.coalesce(
                pair_idx, pair_val, batch.num_nodes, batch.num_nodes, op="add"
            )

        batch.pair_index = pair_idx
        batch.pair_attr = pair_val
        return batch
