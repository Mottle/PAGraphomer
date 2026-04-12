import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym import register
from torch_geometric.utils import to_dense_adj, to_dense_batch


def sinkhorn_transport(cost, eps, n_iters, log_domain=True):
    """Solve entropy-regularized OT plan with uniform marginals.

    Args:
        cost: [N, W, K] cost tensor.
        eps: Entropy regularization parameter (>0).
        n_iters: Number of Sinkhorn iterations (>=0).
        log_domain: Whether to use log-domain for numerical stability.
    Returns:
        transport: [N, W, K] tensor.
    """
    if eps <= 0:
        raise ValueError(f"sinkhorn eps must be > 0, got {eps}")
    if n_iters < 0:
        raise ValueError(f"sinkhorn n_iters must be >= 0, got {n_iters}")
    if n_iters == 0:
        warnings.warn(
            "sinkhorn_iters=0: Sinkhorn algorithm will not perform any iterations. "
            "Transport matrix will be uniform. Set sinkhorn_iters >= 1 for meaningful results."
        )

    n_nodes, n_paths, n_mem = cost.shape
    device = cost.device
    dtype = cost.dtype
    u = torch.full(
        (n_nodes, n_paths), 1.0 / max(n_paths, 1), device=device, dtype=dtype
    )
    v = torch.full((n_nodes, n_mem), 1.0 / max(n_mem, 1), device=device, dtype=dtype)

    if log_domain:
        log_k = -cost / eps
        log_u = torch.log(u + 1e-12)
        log_v = torch.log(v + 1e-12)
        log_a = torch.zeros_like(u)
        log_b = torch.zeros_like(v)
        for _ in range(n_iters):
            log_a = log_u - torch.logsumexp(log_k + log_b.unsqueeze(1), dim=-1)
            log_b = log_v - torch.logsumexp(
                log_k.transpose(1, 2) + log_a.unsqueeze(1), dim=-1
            )
        log_t = log_a.unsqueeze(-1) + log_k + log_b.unsqueeze(1)
        return torch.exp(log_t)

    kernel = torch.exp(-cost / eps)
    a = torch.ones_like(u)
    b = torch.ones_like(v)
    for _ in range(n_iters):
        a = u / (torch.matmul(kernel, b.unsqueeze(-1)).squeeze(-1) + 1e-12)
        b = v / (
            torch.matmul(kernel.transpose(1, 2), a.unsqueeze(-1)).squeeze(-1) + 1e-12
        )
    return a.unsqueeze(-1) * kernel * b.unsqueeze(1)


class OTMotifMemory(nn.Module):
    def __init__(self, dim_h, memory_size):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, dim_h))
        nn.init.xavier_uniform_(self.memory)
        self.proj = nn.Linear(dim_h, dim_h)

    def forward(self, path_repr, sinkhorn_eps, sinkhorn_iters, log_domain=True):
        """Match path features to motif memory.

        Args:
            path_repr: [N, W, D]
        Returns:
            node_motif: [N, D]
            transport: [N, W, K]
            cost: [N, W, K]
        """
        path_norm = F.normalize(path_repr, dim=-1)
        mem_norm = F.normalize(self.memory, dim=-1)
        cost = 1.0 - torch.einsum("nwd,kd->nwk", path_norm, mem_norm)
        transport = sinkhorn_transport(
            cost=cost,
            eps=sinkhorn_eps,
            n_iters=sinkhorn_iters,
            log_domain=log_domain,
        )
        n_paths = path_repr.shape[1]
        # Scale by n_paths so that the weighted sum is normalized
        # per node: without this, nodes with more paths would have
        # proportionally larger motif representations.
        projected = n_paths * torch.einsum("nwk,kd->nwd", transport, self.memory)
        node_motif = projected.mean(dim=1)
        node_motif = self.proj(node_motif)
        return node_motif, transport, cost


class OTFormerBlock(nn.Module):
    """Dual-track update block for node/pair representations."""

    def __init__(
        self,
        dim_h,
        num_heads,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=True,
        use_triangle=True,
        triangle_hidden=32,
        ffn_activation="gelu",
        use_rrwp=False,
        rrwp_dim=0,
    ):
        super().__init__()
        if dim_h % num_heads != 0:
            raise ValueError(
                f"dim_h ({dim_h}) must be divisible by num_heads ({num_heads})."
            )

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.head_dim = dim_h // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_triangle = use_triangle
        self.use_rrwp = bool(use_rrwp)
        self.layer_norm = layer_norm

        self.q_proj = nn.Linear(dim_h, dim_h)
        self.k_proj = nn.Linear(dim_h, dim_h)
        self.v_proj = nn.Linear(dim_h, dim_h)
        self.attn_out = nn.Linear(dim_h, dim_h)
        self.pair_bias = nn.Linear(dim_h, num_heads)
        if self.use_rrwp:
            if rrwp_dim <= 0:
                raise ValueError(
                    f"rrwp_dim must be > 0 when use_rrwp=True, got {rrwp_dim}"
                )
            self.rrwp_bias = nn.Linear(rrwp_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.node_to_pair_u = nn.Linear(dim_h, dim_h)
        self.node_to_pair_v = nn.Linear(dim_h, dim_h)
        self.node_to_pair_out = nn.Linear(dim_h, dim_h)

        if use_triangle:
            self.tri_l = nn.Linear(dim_h, triangle_hidden)
            self.tri_r = nn.Linear(dim_h, triangle_hidden)
            self.tri_out = nn.Linear(triangle_hidden, dim_h)
            self.tri_gate = nn.Linear(dim_h, dim_h)

        act_fn = register.act_dict[ffn_activation]
        self.ffn = nn.Sequential(
            nn.Linear(dim_h, dim_h * 2),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim_h * 2, dim_h),
        )

        if layer_norm:
            self.norm_h1 = nn.LayerNorm(dim_h)
            self.norm_h2 = nn.LayerNorm(dim_h)
            self.norm_z1 = nn.LayerNorm(dim_h)
            self.norm_z2 = nn.LayerNorm(dim_h)
        else:
            self.norm_h1 = nn.Identity()
            self.norm_h2 = nn.Identity()
            self.norm_z1 = nn.Identity()
            self.norm_z2 = nn.Identity()

    def _reshape_heads(self, x):
        # [B, N, D] -> [B, H, N, Dh]
        b, n, _ = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        # [B, H, N, Dh] -> [B, N, D]
        b, h, n, dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(b, n, h * dh)

    def forward(self, h, z, node_mask, rrwp_dense=None):
        node_mask_f = node_mask.unsqueeze(-1).to(h.dtype)
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        pair_mask_f = pair_mask.unsqueeze(-1).to(z.dtype)

        # Keep padded slots clean before any dense interaction.
        h = h * node_mask_f
        z = z * pair_mask_f

        # Pair -> Node biased attention
        h_in = h
        q = self._reshape_heads(self.q_proj(h))
        k = self._reshape_heads(self.k_proj(h))
        v = self._reshape_heads(self.v_proj(h))
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        bias = self.pair_bias(z).permute(0, 3, 1, 2)
        scores = scores + bias
        if self.use_rrwp and rrwp_dense is not None:
            rrwp_bias = self.rrwp_bias(rrwp_dense).permute(0, 3, 1, 2)
            scores = scores + rrwp_bias

        key_valid = node_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        query_valid = node_mask.unsqueeze(1).unsqueeze(3)  # [B,1,N,1]
        scores = scores.masked_fill(~key_valid, -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * query_valid.to(attn.dtype)
        attn = self.attn_dropout(attn)
        h_attn = torch.einsum("bhij,bhjd->bhid", attn, v)
        h_attn = self.attn_out(self._merge_heads(h_attn))
        h = self.norm_h1(h_in + self.dropout(h_attn))
        h = h * node_mask_f

        # Node -> Pair update
        z_in = z
        u = self.node_to_pair_u(h)
        v2 = self.node_to_pair_v(h)
        pair_upd = u.unsqueeze(2) * v2.unsqueeze(1)
        pair_upd = self.node_to_pair_out(pair_upd)
        pair_upd = pair_upd * pair_mask_f
        z = self.norm_z1(z_in + self.dropout(pair_upd))
        z = z * pair_mask_f

        # Pair -> Pair triangle update
        if self.use_triangle:
            left = torch.sigmoid(self.tri_l(z))
            right = self.tri_r(z)
            left = left * pair_mask_f.to(left.dtype)
            right = right * pair_mask_f.to(right.dtype)
            tri = torch.einsum("bikc,bkjc->bijc", left, right) / max(z.shape[1], 1)
            tri = self.tri_out(tri)
            gate = torch.sigmoid(self.tri_gate(z))
            z = self.norm_z2(z + self.dropout(gate * tri))
            z = z * pair_mask_f

        # Node FFN
        h = self.norm_h2(h + self.dropout(self.ffn(h)))
        h = h * node_mask_f

        # Zero invalid padded positions
        h = h * node_mask_f
        z = z * pair_mask_f
        return h, z


def build_pair_init(
    batch,
    dim_h,
    edge_proj,
    pair_init_proj,
    use_spd=True,
    spd_max_dist=16,
    use_rrwp=False,
    rrwp_attr_name="rrwp",
    rrwp_dim=0,
):
    """Build dense pair tensor with bond features and optional SPD bias."""
    node_batch = batch.batch
    h = batch.x
    h_dense, node_mask = to_dense_batch(h, node_batch)
    bsz, nmax, _ = h_dense.shape
    bond_dense = torch.zeros(bsz, nmax, nmax, dim_h, device=h.device, dtype=h.dtype)
    spd_dense = torch.zeros(bsz, nmax, nmax, 1, device=h.device, dtype=h.dtype)
    rrwp_dense = None

    if getattr(batch, "edge_attr", None) is not None:
        edge_emb = edge_proj(batch.edge_attr)
        z_edge = to_dense_adj(
            batch.edge_index, batch=node_batch, edge_attr=edge_emb, max_num_nodes=nmax
        )
        bond_dense = bond_dense + z_edge
    else:
        adj = to_dense_adj(batch.edge_index, batch=node_batch, max_num_nodes=nmax)
        bond_dense[..., :1] = adj.unsqueeze(-1).to(h.dtype)

    if use_spd and hasattr(batch, "ot_spd_index") and hasattr(batch, "ot_spd_val"):
        spd_val = batch.ot_spd_val.to(h.device).to(h.dtype)
        spd_norm = spd_val / float(max(spd_max_dist, 1))
        spd_norm = spd_norm.clamp(min=0.0, max=1.0)
        spd_matrix = to_dense_adj(
            batch.ot_spd_index,
            batch=node_batch,
            edge_attr=spd_norm,
            max_num_nodes=nmax,
        )
        spd_dense = spd_matrix.unsqueeze(-1)

    if use_rrwp:
        rrwp_index_name = f"{rrwp_attr_name}_index"
        rrwp_val_name = f"{rrwp_attr_name}_val"
        if hasattr(batch, rrwp_index_name) and hasattr(batch, rrwp_val_name):
            rrwp_idx = getattr(batch, rrwp_index_name)
            rrwp_val = getattr(batch, rrwp_val_name).to(h.device).to(h.dtype)
            rrwp_dense = to_dense_adj(
                rrwp_idx,
                batch=node_batch,
                edge_attr=rrwp_val,
                max_num_nodes=nmax,
            )
        elif rrwp_dim > 0:
            rrwp_dense = torch.zeros(
                bsz, nmax, nmax, rrwp_dim, device=h.device, dtype=h.dtype
            )
        else:
            raise ValueError(
                "use_rrwp=True requires either precomputed RRWP attributes or rrwp_dim > 0"
            )

    z_in = torch.cat([bond_dense, spd_dense], dim=-1)
    z = pair_init_proj(z_in)
    return z, h_dense, node_mask, rrwp_dense
