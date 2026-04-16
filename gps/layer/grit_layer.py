import warnings

import numpy as np
import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.graphgym.register import act_dict, register_layer
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_add, scatter_max


def pyg_softmax(src, index, num_nodes=None):
    num_nodes = maybe_num_nodes(index, num_nodes)
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out


class MultiHeadAttentionLayerGritSparse(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        use_bias,
        clamp=5.0,
        dropout=0.0,
        act=None,
        edge_enhance=True,
        **kwargs,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1))
        nn.init.xavier_normal_(self.Aw)

        self.act = nn.Identity() if act is None else act_dict[act]()
        if self.edge_enhance:
            self.VeRow = nn.Parameter(
                torch.zeros(self.out_dim, self.num_heads, self.out_dim)
            )
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]
        dest = batch.Q_h[batch.edge_index[1]]
        score = src + dest

        if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            e_w, e_b = batch.E[:, :, : self.out_dim], batch.E[:, :, self.out_dim :]
            score = score * e_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + e_b

        score = self.act(score)
        e_t = score
        if batch.get("E", None) is not None:
            batch.wE = score.flatten(1)

        score = oe.contract("ehd,dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(score, batch.edge_index[1])
        score = self.dropout(score)
        batch.attn = score

        msg = batch.V_h[batch.edge_index[0]] * score
        batch.wV = torch.zeros_like(batch.V_h)
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce="add")

        if self.edge_enhance and batch.E is not None:
            row_v = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
            row_v = oe.contract("nhd,dhc->nhc", row_v, self.VeRow, backend="torch")
            batch.wV = batch.wV + row_v

    def forward(self, batch):
        q_h = self.Q(batch.x)
        k_h = self.K(batch.x)
        v_h = self.V(batch.x)
        batch.E = (
            self.E(batch.edge_attr)
            if batch.get("edge_attr", None) is not None
            else None
        )
        batch.Q_h = q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = k_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = v_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        return batch.wV, batch.get("wE", None)


@register_layer("GritTransformer")
class GritTransformerLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        act="relu",
        norm_e=True,
        O_e=True,
        cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.update_e = cfg.get("update_e", True)
        self.bn_momentum = cfg.get("bn_momentum", 0.1)
        self.bn_no_runner = cfg.get("bn_no_runner", False)
        self.rezero = cfg.get("rezero", False)
        self.act = act_dict[act]() if act is not None else nn.Identity()

        if cfg.get("attn", None) is None:
            cfg.attn = dict()
        self.deg_scaler = cfg.attn.get("deg_scaler", True)

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.0),
            act=cfg.attn.get("act", "relu"),
            edge_enhance=cfg.attn.get("edge_enhance", True),
        )

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        self.O_e = (
            nn.Linear(out_dim // num_heads * num_heads, out_dim)
            if O_e
            else nn.Identity()
        )

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()
            self.layer_norm2_h = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not self.bn_no_runner,
                eps=1e-5,
                momentum=self.bn_momentum,
            )
            self.batch_norm1_e = (
                nn.BatchNorm1d(
                    out_dim,
                    track_running_stats=not self.bn_no_runner,
                    eps=1e-5,
                    momentum=self.bn_momentum,
                )
                if norm_e
                else nn.Identity()
            )
            self.batch_norm2_h = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not self.bn_no_runner,
                eps=1e-5,
                momentum=self.bn_momentum,
            )

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = h
        e_in1 = batch.get("edge_attr", None)
        e = None

        h_attn_out, e_attn_out = self.attention(batch)
        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = self.O_e(
                F.dropout(e_attn_out.flatten(1), self.dropout, training=self.training)
            )

        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
            h = h_in1 + h
            if e is not None:
                if self.rezero:
                    e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None:
                e = self.layer_norm1_e(e)
        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        h_in2 = h
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero:
                h = h * self.alpha2_h
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)
        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        batch.edge_attr = e if self.update_e else e_in1
        return batch


@torch.no_grad()
def get_log_deg(batch):
    if "log_deg" in batch:
        log_deg = batch.log_deg
    elif "deg" in batch:
        log_deg = torch.log(batch.deg + 1).unsqueeze(-1)
    else:
        warnings.warn(
            "Compute degree on the fly; may be inaccurate if edge padding was applied"
        )
        deg = pyg.utils.degree(
            batch.edge_index[1], num_nodes=batch.num_nodes, dtype=torch.float
        )
        log_deg = torch.log(deg + 1)
    return log_deg.view(batch.num_nodes, 1)
