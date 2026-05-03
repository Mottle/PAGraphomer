"""Latent Motif Encoder — RUM-style random walk encoder using GatedDeltaNet (FLA).

Replaces the sequential GRU in RUMLayer with parallel GatedDeltaNet linear
attention for processing random walk sequences.

Output shape: ``(num_samples, N, dim_h)`` — compatible with RUMModel interface
used by OTFormer.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree as pyg_degree

from gps.layer.rum.random_walk import (
    _get_edge_index_and_num_nodes,
    uniform_random_walk,
    uniqueness,
)


class LatentMotifLayer(nn.Module):
    """Single FLA-based layer for random walk motif encoding."""

    def __init__(
        self,
        dim_h,
        num_samples=12,
        rw_length=8,
        num_heads=4,
        gdn_head_dim=16,
        dropout=0.0,
        use_short_conv=False,
    ):
        super().__init__()
        self.dim_h = dim_h
        self.num_samples = num_samples
        self.rw_length = rw_length

        # Sinusoidal walk position encoding.
        pe = self._sinusoidal_pe(rw_length, dim_h)
        self.register_buffer("position_encoding", pe)

        # Project walk features (node + UE + degree) to dim_h.
        self.walk_proj = nn.Linear(dim_h + 3, dim_h)

        # GatedDeltaNet for parallel walk sequence processing.
        from fla.layers import GatedDeltaNet as FLA_GatedDeltaNet

        self.gdn = FLA_GatedDeltaNet(
            hidden_size=dim_h,
            num_heads=num_heads,
            head_dim=gdn_head_dim,
            expand_v=2,
            use_short_conv=use_short_conv,
        )

        self.proj_out = nn.Linear(dim_h, dim_h)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_h)

    @staticmethod
    def _sinusoidal_pe(length, dim):
        pe = torch.zeros(length, dim)
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, g, h):
        """Forward pass.

        Args:
            g: PyG Data-like object with edge_index and num_nodes.
            h: Node features ``(N, dim_h)``.

        Returns:
            path_repr: Walk-level features ``(num_samples, N, dim_h)``.
            loss: scalar (0 for this layer).
        """
        edge_index, num_nodes = _get_edge_index_and_num_nodes(g)

        # 1. Random walk sampling.
        walks, _ = uniform_random_walk(
            g=g, num_samples=self.num_samples, length=self.rw_length
        )
        # walks: (num_samples, N, length)

        # Prepare padded index map (used by both degree and feature gathering).
        pad_idx = torch.full_like(walks, num_nodes)
        walks_safe = torch.where(walks >= 0, walks, pad_idx)

        # 2. Walk uniqueness → direction encoding (sin/cos, like RUM).
        uni = uniqueness(walks)
        uni = uni / max(uni.shape[-1], 1)
        uni = uni * math.pi * 2.0
        uni = torch.stack([uni.sin(), uni.cos()], dim=-1)  # (S, N, L, 2)

        # 3. Node degree (structural signal, same as RUM).
        deg = pyg_degree(edge_index[1], num_nodes=num_nodes, dtype=h.dtype)
        deg = torch.cat([deg, torch.zeros(1, device=deg.device, dtype=deg.dtype)])
        degrees = deg[walks_safe.flatten()].reshape(*walks.shape).unsqueeze(-1)
        max_deg = degrees.max()
        if max_deg > 0:
            degrees = degrees / max_deg
        uni = torch.cat([uni, degrees], dim=-1)  # (S, N, L, 3)

        # 4. Gather node features along walks + position encoding.
        pad_node = torch.zeros(1, self.dim_h, device=h.device, dtype=h.dtype)
        h_padded = torch.cat([h, pad_node], dim=0)

        walk_feat = h_padded[walks_safe]  # (S, N, L, dim_h)
        walk_feat = walk_feat + self.position_encoding  # add PE bias
        walk_feat = torch.cat([walk_feat, uni], dim=-1)  # +UE+deg → (S,N,L, dim_h+3)
        walk_feat = self.walk_proj(walk_feat)  # (S, N, L, dim_h)

        # 5. GatedDeltaNet: process walks in parallel.
        S, N, L, D = walk_feat.shape
        walk_flat = walk_feat.reshape(S * N, L, D)  # (S*N, L, D) batch of sequences
        gdn_out, *_ = self.gdn(walk_flat)  # (S*N, L, D)
        gdn_out = gdn_out.reshape(S, N, L, D)  # (S, N, L, D)

        # 6. Aggregate to per-walk features.
        h_out = gdn_out.mean(dim=2)  # (S, N, D)
        h_out = self.norm(h_out + self.dropout(self.proj_out(h_out)))

        return h_out, 0.0


class LatentMotifModel(nn.Module):
    """Drop-in replacement for RUMModel.

    Uses LatentMotifLayer (GatedDeltaNet) instead of RUMLayer (GRU).
    Output interface matches RUMModel exactly: ``(num_samples, N, dim_h)``.
    """

    def __init__(
        self,
        dim_h,
        num_samples=12,
        rw_length=8,
        num_heads=4,
        gdn_head_dim=16,
        dropout=0.0,
        use_short_conv=False,
    ):
        super().__init__()
        self.dim_h = dim_h
        self.fc_in = nn.Linear(dim_h, dim_h)
        self.fc_out = nn.Linear(dim_h, dim_h)
        self.layer = LatentMotifLayer(
            dim_h=dim_h,
            num_samples=num_samples,
            rw_length=rw_length,
            num_heads=num_heads,
            gdn_head_dim=gdn_head_dim,
            dropout=dropout,
            use_short_conv=use_short_conv,
        )

    def forward(self, g, h, e=None):
        """Forward pass.

        Args:
            g: PyG Data-like object with edge_index.
            h: Node features ``(N, dim_h)``.
            e: Edge features (ignored, for RUM compatibility).

        Returns:
            path_repr: ``(num_samples, N, dim_h)`` — walk-level representations.
            loss: scalar (0).
        """
        h = self.fc_in(h)
        h, _loss = self.layer(g, h)
        h = self.fc_out(h)
        return h, _loss
