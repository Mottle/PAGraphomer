import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import GatedDeltaNet as FLA_GatedDeltaNet
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import degree

from gps.head.san_graph import SANGraphHead


class GatedDeltaNetPEEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc_x = nn.Linear(dim_in, 16, bias=False)
        self.fc_rwse = nn.Linear(16, 6, bias=False)
        self.fc_lap = nn.Linear(12, 6, bias=False)
        self.fc_deg = nn.Linear(1, 4, bias=False)

    def forward(self, batch):
        N = batch.x.size(0)

        x_emb = self.fc_x(batch.x)
        deg = degree(batch.edge_index[0], num_nodes=N).float().view(-1, 1)
        deg_emb = self.fc_deg(deg)

        rwse = getattr(batch, "rwse", None)
        if rwse is None or rwse.size(1) != 16:
            rwse = torch.zeros(N, 16, device=x_emb.device)
        rwse_emb = self.fc_rwse(rwse)

        eigvecs = getattr(batch, "eigvecs", None)
        if eigvecs is None or eigvecs.size(1) != 12:
            eigvecs = torch.zeros(N, 12, device=x_emb.device)
        lappe_emb = self.fc_lap(eigvecs)

        batch.x = torch.cat([x_emb, rwse_emb, lappe_emb, deg_emb], dim=-1)
        return batch


class GatedDeltaNetLayer(nn.Module):
    def __init__(self, dim_h, dropout=0.0, num_heads=4):
        super().__init__()
        self.gdn = FLA_GatedDeltaNet(
            hidden_size=dim_h,
            num_heads=num_heads,
            head_dim=16,
            expand_v=2,
            use_short_conv=False,
        )
        self.norm1 = nn.LayerNorm(dim_h)
        self.norm2 = nn.LayerNorm(dim_h)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim_h, dim_h * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_h * 4, dim_h),
        )

    def forward(self, h):
        N, H = h.shape
        out, *_ = self.gdn(h.unsqueeze(0))
        out = out.squeeze(0)
        h = self.norm1(h + out)
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

        P = max(getattr(cfg.gt, "perm_ensemble", 1), 1)
        pct = getattr(cfg.gt, "perm_pct", 20)
        if self.training and P > 1:
            all_logits = []
            for _ in range(P):
                h = self._perturb(batch.x, pct)
                for layer in self.layers:
                    h = layer(h)
                batch.x = h
                pred, _ = self.head(batch)
                all_logits.append(pred)
            logits = torch.stack(all_logits, dim=0)
            pred = logits.mean(dim=0)
            self._consistency_loss = logits.var(dim=0).mean() * cfg.gt.perm_lambda
            return pred, batch.y
        else:
            self._consistency_loss = torch.tensor(0.0, device=batch.x.device)
            for layer in self.layers:
                batch.x = layer(batch.x)
            return self.head(batch)
