import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_geometric.graphgym.register import act_dict, register_layer
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_scatter import scatter, scatter_add, scatter_max


def pyg_softmax(src, index, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(index.max().item()) + 1 if index.numel() > 0 else 0
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out


def _groupwise_sparsemax(src, index, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(index.max().item()) + 1 if index.numel() > 0 else 0
    out = torch.zeros_like(src)
    for gid in range(num_nodes):
        mask = index == gid
        if not mask.any():
            continue
        z = src[mask]
        z_sorted, _ = torch.sort(z, dim=0, descending=True)
        z_cumsum = z_sorted.cumsum(dim=0)
        ks = torch.arange(1, z.size(0) + 1, device=src.device, dtype=src.dtype).view(
            -1, 1
        )
        support = 1 + ks * z_sorted > z_cumsum
        k_z = support.sum(dim=0).clamp(min=1)
        tau_sum = z_cumsum.gather(0, (k_z - 1).unsqueeze(0)).squeeze(0)
        tau = (tau_sum - 1.0) / k_z.to(src.dtype)
        probs = torch.clamp(z - tau.unsqueeze(0), min=0.0)
        out[mask] = probs
    return out


def _groupwise_entmax15(src, index, num_nodes=None, n_iter=25):
    if num_nodes is None:
        num_nodes = int(index.max().item()) + 1 if index.numel() > 0 else 0
    out = torch.zeros_like(src)
    power = 2.0
    for gid in range(num_nodes):
        mask = index == gid
        if not mask.any():
            continue
        z = src[mask]
        max_z = z.max(dim=0, keepdim=True).values
        tau_lo = max_z - 1.0
        tau_hi = max_z
        for _ in range(n_iter):
            tau_mid = (tau_lo + tau_hi) / 2.0
            p = torch.clamp(z - tau_mid, min=0.0) ** power
            sums = p.sum(dim=0, keepdim=True)
            tau_lo = torch.where(sums > 1.0, tau_mid, tau_lo)
            tau_hi = torch.where(sums > 1.0, tau_hi, tau_mid)
        probs = torch.clamp(z - tau_hi, min=0.0) ** power
        probs = probs / probs.sum(dim=0, keepdim=True).clamp_min(1e-16)
        out[mask] = probs
    return out


def _dense_sparsemax(src, dim=-1):
    z_sorted, _ = torch.sort(src, dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim=dim)
    dims = src.size(dim)
    shape = [1] * src.dim()
    shape[dim] = dims
    ks = torch.arange(1, dims + 1, device=src.device, dtype=src.dtype).view(shape)
    support = 1 + ks * z_sorted > z_cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau_sum = z_cumsum.gather(dim, k_z - 1)
    tau = (tau_sum - 1.0) / k_z.to(src.dtype)
    return torch.clamp(src - tau, min=0.0)


def _dense_entmax_bisect(src, alpha, dim=-1, n_iter=25):
    if alpha <= 1.0 + 1e-8:
        return torch.softmax(src, dim=dim)
    if alpha >= 2.0 - 1e-8:
        probs = _dense_sparsemax(src, dim=dim)
        return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-16)

    power = 1.0 / (alpha - 1.0)
    max_val = src.max(dim=dim, keepdim=True).values
    tau_lo = max_val - 1.0
    tau_hi = max_val
    for _ in range(n_iter):
        tau_mid = (tau_lo + tau_hi) / 2.0
        probs = torch.clamp(src - tau_mid, min=0.0) ** power
        sums = probs.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(sums > 1.0, tau_mid, tau_lo)
        tau_hi = torch.where(sums > 1.0, tau_hi, tau_mid)
    probs = torch.clamp(src - tau_hi, min=0.0) ** power
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-16)


def grouped_normalize(src, index, kind="softmax", num_nodes=None):
    if kind == "softmax":
        return pyg_softmax(src, index, num_nodes=num_nodes)
    if kind == "sparsemax":
        return _groupwise_sparsemax(src, index, num_nodes=num_nodes)
    if kind == "entmax15":
        return _groupwise_entmax15(src, index, num_nodes=num_nodes)
    raise ValueError(f"Unsupported attention normalizer: {kind}")


def dense_alpha_normalize(src, alpha, dim=-1):
    return _dense_entmax_bisect(src, alpha=alpha, dim=dim)


def _cfg_get(cfg, name, default):
    if cfg is None:
        return default
    if hasattr(cfg, name):
        return getattr(cfg, name)
    if hasattr(cfg, "get"):
        return cfg.get(name, default)
    return default


class ScaledRangeFormerPairScore(nn.Module):
    def __init__(
        self,
        dim_h,
        num_heads,
        use_bias=False,
        clamp=5.0,
        act="relu",
    ):
        super().__init__()
        if dim_h % num_heads != 0:
            raise ValueError(
                f"dim_h ({dim_h}) must be divisible by num_heads ({num_heads})"
            )

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.head_dim = dim_h // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.clamp = abs(clamp) if clamp is not None else None
        self.act = act_dict[act]() if act is not None else nn.Identity()

        self.Q = nn.Linear(dim_h, dim_h, bias=use_bias)
        self.K = nn.Linear(dim_h, dim_h, bias=use_bias)
        self.V = nn.Linear(dim_h, dim_h, bias=use_bias)
        self.E = nn.Linear(dim_h, dim_h, bias=True)
        self.edge_value = nn.Linear(dim_h, dim_h, bias=use_bias)

        self.attn_vec = nn.Parameter(torch.zeros(1, num_heads, self.head_dim))

        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.edge_value.weight)
        nn.init.xavier_uniform_(self.attn_vec)

        if self.Q.bias is not None:
            nn.init.zeros_(self.Q.bias)
        if self.K.bias is not None:
            nn.init.zeros_(self.K.bias)
        if self.V.bias is not None:
            nn.init.zeros_(self.V.bias)
        if self.E.bias is not None:
            nn.init.zeros_(self.E.bias)
        if self.edge_value.bias is not None:
            nn.init.zeros_(self.edge_value.bias)

    def forward(self, h, pair_attr, pair_index):
        src, dst = pair_index[0], pair_index[1]

        q = self.Q(h).view(-1, self.num_heads, self.head_dim)
        k = self.K(h).view(-1, self.num_heads, self.head_dim)
        v = self.V(h).view(-1, self.num_heads, self.head_dim)
        e = self.E(pair_attr).view(-1, self.num_heads, self.head_dim)

        q_dst = q[dst]
        k_src = k[src]
        v_src = v[src]

        relation = self.act(q_dst + k_src + e)
        base_score = (q_dst * k_src).sum(dim=-1, keepdim=True) / self.scale
        edge_bias = (relation * self.attn_vec).sum(dim=-1, keepdim=True)
        score = base_score + edge_bias
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        edge_msg = self.edge_value(pair_attr).view(-1, self.num_heads, self.head_dim)
        value = v_src + relation + edge_msg
        return score, value, relation


class ScaledRangeFormerAttention(nn.Module):
    NEG_INF = -1e9

    def __init__(self, dim_h, num_heads, attn_dropout=0.0, formulation="B", cfg=None):
        super().__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.formulation = formulation
        self.dropout = nn.Dropout(attn_dropout)

        ms_cfg = _cfg_get(cfg, "msrrwp", None)
        self.rrwp_name = str(_cfg_get(ms_cfg, "rrwp_name", "rrwp"))
        self.spd_name = str(_cfg_get(ms_cfg, "spd_name", "srf_spd"))
        self.use_spd = bool(_cfg_get(ms_cfg, "use_spd", True))
        self.scale_mode = str(_cfg_get(ms_cfg, "scale_mode", "percentile"))
        self.mask_mode = str(_cfg_get(ms_cfg, "mask_mode", "hard"))
        self.soft_eps = float(_cfg_get(ms_cfg, "soft_eps", 1e-6))
        self.hard_eps = float(_cfg_get(ms_cfg, "hard_eps", 1e-9))
        self.weight_in_softmax = bool(_cfg_get(ms_cfg, "weight_in_softmax", False))
        self.learnable_weights = bool(_cfg_get(ms_cfg, "learnable_weights", True))
        self.inner_residual = bool(_cfg_get(ms_cfg, "inner_residual", True))
        self.inner_norm = bool(_cfg_get(ms_cfg, "inner_norm", True))
        self.thresholds = [
            float(v) for v in list(_cfg_get(ms_cfg, "thresholds", [1.0]))
        ]
        if not self.thresholds:
            self.thresholds = [1.0]
        self.num_scales = len(self.thresholds)
        self.alphas = [float(v) for v in list(_cfg_get(ms_cfg, "alphas", [1.0]))]
        if len(self.alphas) == 1 and self.num_scales > 1:
            self.alphas = self.alphas * self.num_scales
        if len(self.alphas) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} scale alphas, got {len(self.alphas)}."
            )

        attn_cfg = _cfg_get(cfg, "attn", None)
        use_bias = bool(_cfg_get(attn_cfg, "use_bias", False))
        clamp = float(_cfg_get(attn_cfg, "clamp", 5.0))
        act = str(_cfg_get(attn_cfg, "act", "relu"))

        if self.formulation in ["A", "B"]:
            self.shared_score = ScaledRangeFormerPairScore(
                dim_h=dim_h,
                num_heads=num_heads,
                use_bias=use_bias,
                clamp=clamp,
                act=act,
            )
        else:
            self.score_layers = nn.ModuleList(
                [
                    ScaledRangeFormerPairScore(
                        dim_h=dim_h,
                        num_heads=num_heads,
                        use_bias=use_bias,
                        clamp=clamp,
                        act=act,
                    )
                    for _ in range(self.num_scales)
                ]
            )
            self.inner_node_norms = nn.ModuleList(
                [nn.LayerNorm(dim_h) for _ in range(self.num_scales)]
            )
            self.inner_edge_norms = nn.ModuleList(
                [nn.LayerNorm(dim_h) for _ in range(self.num_scales)]
            )

        init = torch.zeros(self.num_scales)
        if self.learnable_weights:
            self.scale_logits = nn.Parameter(init)
        else:
            self.register_buffer("scale_logits", init, persistent=False)

    def _coalesce_pair_graph(self, batch):
        pair_index = batch.get("pair_index", None)
        pair_attr = batch.get("pair_attr", None)
        if pair_index is None:
            raise ValueError(
                f"{self.__class__.__name__} requires batch.pair_index. "
                f"Prepare pair states before entering this layer."
            )
        if pair_attr is None:
            raise ValueError(
                f"{self.__class__.__name__} requires pair features in batch.pair_attr."
            )
        return pair_index, pair_attr

    def _scale_weights(self):
        return torch.softmax(self.scale_logits, dim=0)

    def _build_pair_graph_ids(self, batch, pair_index):
        node_graph = batch.batch.to(pair_index.device)
        src_graph = node_graph[pair_index[0]]
        dst_graph = node_graph[pair_index[1]]
        if not torch.equal(src_graph, dst_graph):
            raise ValueError(
                f"{self.__class__.__name__} received cross-graph pair indices, which is invalid."
            )
        return src_graph

    def _precomputed_pair_data(self, batch, pair_index):
        pre_idx = getattr(batch, "srf_pair_index", None)
        pre_mask = getattr(batch, "srf_mask_val", None)
        if pre_idx is None or pre_mask is None:
            return None, None
        if pre_idx.shape != pair_index.shape or not torch.equal(pre_idx, pair_index):
            return None, None
        return pre_idx, pre_mask.to(pair_index.device)

    def _align_spd(self, batch, pair_index, pair_attr):
        if not self.use_spd:
            raise ValueError(f"{self.__class__.__name__} expects use_spd=True.")

        spd_index_name = f"{self.spd_name}_index"
        spd_val_name = f"{self.spd_name}_val"
        if not hasattr(batch, spd_index_name) or not hasattr(batch, spd_val_name):
            raise ValueError(
                f"{self.__class__.__name__} requires precomputed SPD pair features "
                f"('{spd_index_name}', '{spd_val_name}')."
            )

        spd_idx = getattr(batch, spd_index_name)
        spd_val = getattr(batch, spd_val_name).to(pair_attr.device).to(torch.float)
        pre_idx, pre_mask = self._precomputed_pair_data(batch, pair_index)
        if pre_idx is not None:
            return spd_val.to(pair_attr.device)
        zeros = spd_val.new_zeros(pair_index.size(1))
        aligned_index, aligned_spd = torch_sparse.coalesce(
            torch.cat([pair_index, spd_idx], dim=1),
            torch.cat([zeros, spd_val], dim=0),
            batch.num_nodes,
            batch.num_nodes,
            op="add",
        )
        if aligned_index.size(1) != pair_index.size(1):
            raise RuntimeError(
                "SPD alignment changed pair graph cardinality unexpectedly."
            )
        return aligned_spd

    def _compute_scale_values(self, batch, spd_full, pair_index):
        _, pre_mask = self._precomputed_pair_data(batch, pair_index)
        if pre_mask is not None:
            if pre_mask.size(1) != self.num_scales:
                raise ValueError(
                    f"Precomputed mask scale count ({pre_mask.size(1)}) does not match expected "
                    f"count ({self.num_scales})."
                )
            return [
                pre_mask[:, idx].to(spd_full.dtype) for idx in range(pre_mask.size(1))
            ]

        if len(self.thresholds) != self.num_scales:
            raise ValueError(
                f"Configured thresholds ({len(self.thresholds)}) do not match initialized "
                f"scale count ({self.num_scales})."
            )

        src, dst = pair_index[0], pair_index[1]
        is_self = src == dst
        pair_graph = self._build_pair_graph_ids(batch, pair_index)
        num_graphs = int(pair_graph.max().item()) + 1 if pair_graph.numel() > 0 else 0

        scale_values = [
            torch.zeros_like(spd_full, dtype=spd_full.dtype) for _ in self.thresholds
        ]
        for gid in range(num_graphs):
            graph_mask = pair_graph == gid
            graph_non_self = graph_mask & (~is_self)
            graph_spd = spd_full[graph_non_self]

            if graph_spd.numel() == 0:
                cutoffs = [spd_full.new_tensor(0.0) for _ in self.thresholds]
            else:
                sorted_spd = torch.sort(graph_spd).values
                cutoffs = []
                total = sorted_spd.numel()
                for theta in self.thresholds:
                    theta = min(max(float(theta), 0.0), 1.0)
                    idx = min(total - 1, max(0, int(math.ceil(theta * total)) - 1))
                    cutoffs.append(sorted_spd[idx])

            graph_self = graph_mask & is_self
            for idx, cutoff in enumerate(cutoffs):
                active = graph_mask & (spd_full <= cutoff)
                active = torch.logical_or(active, graph_self)
                scale_values[idx][active] = 1.0
        return scale_values

    def _apply_structure(self, score, scale_values):
        coeff = scale_values.to(score.dtype)
        if self.mask_mode == "soft":
            masked_score = score * coeff.unsqueeze(-1).unsqueeze(-1)
            edge_coeff = coeff
        else:
            active = coeff > self.hard_eps
            masked_score = score.masked_fill(
                ~active.unsqueeze(-1).unsqueeze(-1), self.NEG_INF
            )
            edge_coeff = active.to(score.dtype)
        return masked_score, edge_coeff

    def _aggregate(self, pair_index, score, value, relation, scale_values):
        dst = pair_index[1]
        num_nodes = int(dst.max().item()) + 1 if dst.numel() > 0 else 0
        score, edge_coeff = self._apply_structure(score, scale_values)
        attn = grouped_normalize(score, dst, kind="softmax", num_nodes=num_nodes)
        attn = self.dropout(attn)
        msg = value * attn
        node_out = torch.zeros_like(
            value.new_zeros(num_nodes, value.size(1), value.size(2))
        )
        scatter(msg, dst, dim=0, out=node_out, reduce="add")
        edge_out = relation * edge_coeff.unsqueeze(-1).unsqueeze(-1)
        return node_out.flatten(1), edge_out.flatten(1)

    def _dense_aggregate_alpha(
        self, batch, pair_index, score, value, relation, scale_values, alpha
    ):
        node_batch = batch.batch
        _, node_mask = to_dense_batch(batch.x, node_batch)
        bsz, nmax = node_mask.shape
        num_heads = score.size(1)
        head_dim = value.size(2)

        # We aggregate messages from src -> dst, so build dense matrices indexed as
        # [dst, src] to normalize along the source dimension.
        pair_index_ds = torch.stack([pair_index[1], pair_index[0]], dim=0)

        dense_score = to_dense_adj(
            pair_index_ds,
            batch=node_batch,
            edge_attr=score.squeeze(-1),
            max_num_nodes=nmax,
        ).permute(0, 3, 1, 2)

        dense_value = to_dense_adj(
            pair_index_ds,
            batch=node_batch,
            edge_attr=value.reshape(value.size(0), num_heads * head_dim),
            max_num_nodes=nmax,
        ).view(bsz, nmax, nmax, num_heads, head_dim)
        dense_value = dense_value.permute(0, 3, 1, 2, 4)

        dense_scale = to_dense_adj(
            pair_index_ds,
            batch=node_batch,
            edge_attr=scale_values.to(score.dtype),
            max_num_nodes=nmax,
        )

        pair_valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        pair_valid = pair_valid.unsqueeze(1)

        if self.mask_mode == "soft":
            dense_score = dense_score * dense_scale.unsqueeze(1)
            dense_score = dense_score.masked_fill(~pair_valid, self.NEG_INF)
            edge_coeff = scale_values.to(relation.dtype)
        else:
            active = (dense_scale > self.hard_eps).unsqueeze(1) & pair_valid
            dense_score = dense_score.masked_fill(~active, self.NEG_INF)
            edge_coeff = (scale_values > self.hard_eps).to(relation.dtype)

        attn = dense_alpha_normalize(dense_score, alpha=alpha, dim=-1)
        attn = self.dropout(attn)
        out_dense = torch.einsum("bhij,bhijd->bhid", attn, dense_value)
        out_dense = out_dense.permute(0, 2, 1, 3)
        node_out = out_dense[node_mask]

        edge_out = relation * edge_coeff.unsqueeze(-1).unsqueeze(-1)
        return node_out.flatten(1), edge_out.flatten(1)

    def _formulation_a(self, pair_index, score, value, relation, scale_values):
        weights = self._scale_weights()
        coeff = sum(w * scale for w, scale in zip(weights, scale_values))
        if self.mask_mode == "soft":
            attn_score = score * coeff.unsqueeze(-1).unsqueeze(-1)
            edge_coeff = coeff
        else:
            active = coeff > 0
            attn_score = (score * coeff.unsqueeze(-1).unsqueeze(-1)).masked_fill(
                ~active.unsqueeze(-1).unsqueeze(-1), self.NEG_INF
            )
            edge_coeff = active.to(score.dtype)
        dst = pair_index[1]
        num_nodes = int(dst.max().item()) + 1 if dst.numel() > 0 else 0
        alpha = grouped_normalize(attn_score, dst, kind="softmax", num_nodes=num_nodes)
        alpha = self.dropout(alpha)
        msg = value * alpha
        node_out = torch.zeros(
            num_nodes, value.size(1), value.size(2), device=value.device
        )
        scatter(msg, dst, dim=0, out=node_out, reduce="add")
        edge_out = relation * edge_coeff.unsqueeze(-1).unsqueeze(-1)
        return node_out.flatten(1), edge_out.flatten(1)

    def _formulation_b(self, batch, pair_index, score, value, relation, scale_values):
        weights = self._scale_weights()
        node_mix = None
        edge_mix = None
        num_scales = len(scale_values)
        for idx, scale in enumerate(scale_values):
            scale_score = score
            if self.weight_in_softmax:
                scale_score = scale_score * weights[idx]
                mix_weight = 1.0 / num_scales
            else:
                mix_weight = weights[idx]
            alpha = self.alphas[idx]
            if alpha > 1.0 + 1e-8:
                node_out, edge_out = self._dense_aggregate_alpha(
                    batch, pair_index, scale_score, value, relation, scale, alpha
                )
            else:
                node_out, edge_out = self._aggregate(
                    pair_index,
                    scale_score,
                    value,
                    relation,
                    scale,
                )
            node_out = node_out * mix_weight
            edge_out = edge_out * mix_weight
            node_mix = node_out if node_mix is None else node_mix + node_out
            edge_mix = edge_out if edge_mix is None else edge_mix + edge_out
        return node_mix, edge_mix

    def _sequential_update(
        self, batch, h_cur, e_cur, pair_index, score_layer, scale_values, alpha
    ):
        score, value, relation = score_layer(h_cur, e_cur, pair_index)
        if alpha > 1.0 + 1e-8:
            return self._dense_aggregate_alpha(
                batch, pair_index, score, value, relation, scale_values, alpha
            )
        return self._aggregate(pair_index, score, value, relation, scale_values)

    def _formulation_c_or_d(self, batch, h, pair_attr, pair_index, scale_values):
        h_cur = h
        e_cur = pair_attr
        h_states = []
        e_states = []
        for idx, scale in enumerate(scale_values):
            h_next, e_next = self._sequential_update(
                batch,
                h_cur,
                e_cur,
                pair_index,
                self.score_layers[idx],
                scale,
                self.alphas[idx],
            )
            if self.inner_residual:
                h_cur = h_cur + h_next
                e_cur = e_cur + e_next
            else:
                h_cur = h_next
                e_cur = e_next
            if self.inner_norm:
                h_cur = self.inner_node_norms[idx](h_cur)
                e_cur = self.inner_edge_norms[idx](e_cur)
            if self.formulation == "D":
                h_states.append(h_cur)
                e_states.append(e_cur)

        if self.formulation == "C":
            return h_cur, e_cur

        weights = self._scale_weights()
        h_mix = None
        e_mix = None
        for idx, weight in enumerate(weights):
            h_part = h_states[idx] * weight
            e_part = e_states[idx] * weight
            h_mix = h_part if h_mix is None else h_mix + h_part
            e_mix = e_part if e_mix is None else e_mix + e_part
        return h_mix, e_mix

    def forward(self, batch):
        pair_index, pair_attr = self._coalesce_pair_graph(batch)
        spd_full = self._align_spd(batch, pair_index, pair_attr)
        scale_values = self._compute_scale_values(batch, spd_full, pair_index)

        if self.formulation in ["A", "B"]:
            score, value, relation = self.shared_score(batch.x, pair_attr, pair_index)
            if self.formulation == "A":
                h_out, e_out = self._formulation_a(
                    pair_index, score, value, relation, scale_values
                )
            else:
                h_out, e_out = self._formulation_b(
                    batch, pair_index, score, value, relation, scale_values
                )
        else:
            h_out, e_out = self._formulation_c_or_d(
                batch, batch.x, pair_attr, pair_index, scale_values
            )

        return h_out, e_out


class ScaledRangeFormerLayer(nn.Module):
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
        formulation="B",
        **kwargs,
    ):
        super().__init__()
        if in_dim != out_dim:
            raise ValueError(
                f"ScaledRangeFormerLayer expects in_dim == out_dim, got "
                f"{in_dim} and {out_dim}."
            )

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.update_e = _cfg_get(cfg, "update_e", True)
        self.bn_momentum = float(_cfg_get(cfg, "bn_momentum", 0.1))
        self.bn_no_runner = bool(_cfg_get(cfg, "bn_no_runner", False))
        self.rezero = bool(_cfg_get(cfg, "rezero", False))
        self.act = act_dict[act]() if act is not None else nn.Identity()

        self.attention = ScaledRangeFormerAttention(
            dim_h=out_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            formulation=formulation,
            cfg=cfg,
        )

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim) if O_e else nn.Identity()

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
        h_in1 = batch.x
        e_in1 = batch.get("pair_attr", None)

        h_attn_out, e_attn_out = self.attention(batch)
        h = self.O_h(F.dropout(h_attn_out, self.dropout, training=self.training))
        e = self.O_e(F.dropout(e_attn_out, self.dropout, training=self.training))

        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
                e = e * self.alpha1_e
            h = h_in1 + h
            if e_in1 is not None:
                e = e_in1 + e

        if self.layer_norm:
            h = self.layer_norm1_h(h)
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
        batch.pair_attr = e if self.update_e else e_in1
        return batch


@register_layer("ScaledRangeFormerShared")
class ScaledRangeFormerSharedLayer(ScaledRangeFormerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, formulation="A", **kwargs)


@register_layer("ScaledRangeFormerMixed")
class ScaledRangeFormerMixedLayer(ScaledRangeFormerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, formulation="B", **kwargs)


@register_layer("ScaledRangeFormerSequential")
class ScaledRangeFormerSequentialLayer(ScaledRangeFormerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, formulation="C", **kwargs)


@register_layer("ScaledRangeFormerBypass")
class ScaledRangeFormerBypassLayer(ScaledRangeFormerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, formulation="D", **kwargs)
