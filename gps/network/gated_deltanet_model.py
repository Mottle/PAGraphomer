import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import GatedDeltaNet as FLA_GatedDeltaNet
from torch_geometric.graphgym import cfg, register
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import degree
from torch_scatter import scatter_add

from gps.head.san_graph import SANGraphHead


class GatedDeltaNetPEEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_x = getattr(cfg.gt, "pe_dim_x", 16)
        self.dim_rwse = getattr(cfg.gt, "pe_dim_rwse", 6)
        self.dim_lap = getattr(cfg.gt, "pe_dim_lap", 6)
        self.dim_deg = getattr(cfg.gt, "pe_dim_deg", 4)
        self.total_dim = self.dim_x + self.dim_rwse + self.dim_lap + self.dim_deg

        self.fc_x = nn.Linear(dim_in, self.dim_x, bias=False)
        self.num_rwse_steps = (
            len(cfg.posenc_RWSE.kernel.times) if cfg.posenc_RWSE.enable else 0
        )
        self.fc_rwse = (
            nn.Linear(self.num_rwse_steps, self.dim_rwse, bias=False)
            if self.num_rwse_steps > 0
            else nn.Identity()
        )
        self.fc_lap = nn.Linear(12, self.dim_lap, bias=False)
        self.fc_deg = nn.Linear(1, self.dim_deg, bias=False)

    def forward(self, batch):
        N = batch.x.size(0)

        x_emb = self.fc_x(batch.x)
        deg = degree(batch.edge_index[0], num_nodes=N).float().view(-1, 1)
        deg_emb = self.fc_deg(deg)

        rwse = getattr(batch, "pestat_RWSE", None)
        if rwse is None or rwse.size(1) != self.num_rwse_steps:
            rwse = torch.zeros(N, self.num_rwse_steps, device=x_emb.device)
        rwse_emb = self.fc_rwse(rwse)

        eigvecs = getattr(batch, "EigVecs", None)
        if eigvecs is None or eigvecs.size(1) != 12:
            eigvecs = torch.zeros(N, 12, device=x_emb.device)
        elif torch.isnan(eigvecs).any() or torch.isinf(eigvecs).any():
            eigvecs = torch.zeros(N, 12, device=x_emb.device)
        lappe_emb = self.fc_lap(eigvecs)

        batch.x = torch.cat([x_emb, rwse_emb, lappe_emb, deg_emb], dim=-1)
        return batch


class GatedDeltaNetLayer(nn.Module):
    def __init__(self, dim_h, dropout=0.0, num_heads=4):
        super().__init__()
        use_sc = getattr(cfg.gt, "gdn_short_conv", False)
        head_dim = getattr(cfg.gt, "gdn_head_dim", 16)
        expand_v = getattr(cfg.gt, "gdn_expand_v", 2)
        fla_kw = dict(
            hidden_size=dim_h,
            num_heads=num_heads,
            head_dim=head_dim,
            expand_v=expand_v,
            use_short_conv=use_sc,
        )
        self.gdn_global = FLA_GatedDeltaNet(**fla_kw)
        self.gdn_struct = FLA_GatedDeltaNet(**fla_kw)
        self.neighbor_proj = nn.Linear(dim_h, dim_h, bias=False)

        self.edge_to_node_proj = nn.Linear(dim_h, dim_h, bias=False)

        self.norm1 = nn.LayerNorm(dim_h)
        self.norm2 = nn.LayerNorm(dim_h)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim_h, dim_h * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_h * 4, dim_h),
        )

    def forward(self, h, edge_index, edge_attr=None):
        N, D = h.shape
        h_in = h
        src, dst = edge_index

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1).float()
            node_edge_feat = scatter_add(
                edge_attr.expand(-1, D) if edge_attr.size(1) == 1 else edge_attr,
                dst,
                dim=0,
                dim_size=N,
            )
            h = h + self.edge_to_node_proj(node_edge_feat)

        neighbor_msg = self.neighbor_proj(h[src])
        h_neighbor = scatter_add(neighbor_msg, dst, dim=0, dim_size=N)
        deg = scatter_add(
            torch.ones(edge_index.size(1), device=h.device), dst, dim_size=N
        ).view(-1, 1)
        h_neighbor = h_neighbor / (deg + 1e-8)
        h_struct, *_ = self.gdn_struct(h_neighbor.unsqueeze(0))
        h_struct = h_struct.squeeze(0)

        h_global, *_ = self.gdn_global(h.unsqueeze(0))
        h_global = h_global.squeeze(0)

        h = self.norm1(h_in + h_struct + h_global)
        h = self.norm2(h + self.ffn(h))
        return h


@register_network("GatedDeltaNetModel")
class GatedDeltaNetModel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_h = cfg.gt.dim_hidden
        num_layers = cfg.gt.layers
        dropout = float(getattr(cfg.gt, "dropout", 0.1))
        n_heads = cfg.gt.n_heads

        self.encoder = GatedDeltaNetPEEncoder(dim_in, dim_h)
        self.total_pe_dim = self.encoder.total_dim
        self.proj = (
            nn.Identity()
            if self.total_pe_dim == dim_h
            else nn.Linear(self.total_pe_dim, dim_h)
        )

        self.edge_encoder = None
        self.edge_dim = 0
        if cfg.dataset.edge_encoder:
            if "PNA" in cfg.gt.layer_type:
                self.edge_dim = min(128, cfg.gnn.dim_inner)
            else:
                self.edge_dim = cfg.gnn.dim_inner
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(self.edge_dim)

        self.layers = nn.ModuleList(
            [
                GatedDeltaNetLayer(dim_h, dropout=dropout, num_heads=n_heads)
                for _ in range(num_layers)
            ]
        )
        self.head = SANGraphHead(
            dim_in=dim_h,
            dim_out=dim_out,
            L=cfg.gnn.layers_post_mp,
        )

    def _perturb(self, x, pct):
        noise = torch.rand(x.size(0), device=x.device) * pct / 100
        idx = torch.argsort(torch.arange(x.size(0), device=x.device).float() + noise)
        return x[idx]

    def forward(self, batch):
        batch = self.encoder(batch)
        batch.x = self.proj(batch.x)

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None
        if self.edge_encoder is not None:
            batch = self.edge_encoder(batch)
            edge_attr = batch.edge_attr

        P = max(getattr(cfg.gt, "perm_ensemble", 1), 1)
        pct = getattr(cfg.gt, "perm_pct", 20)
        if self.training and P > 1 and edge_attr is not None:
            P = 1

        if self.training and P > 1:
            all_logits = []
            for _ in range(P):
                h = self._perturb(batch.x, pct)
                for layer in self.layers:
                    h = layer(h, edge_index, edge_attr)
                batch.x = h
                pred, _ = self.head(batch)
                all_logits.append(pred)
            logits = torch.stack(all_logits, dim=0)
            pred = logits.mean(dim=0)
            self._consistency_loss = logits.var(dim=0).mean() * cfg.gt.perm_lambda
            return pred, batch.y
        else:
            self._consistency_loss = torch.tensor(0.0, device=batch.x.device)
            h = batch.x
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr)
            batch.x = h
            return self.head(batch)
