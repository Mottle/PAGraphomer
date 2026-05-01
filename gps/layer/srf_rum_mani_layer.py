import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gps.layer.rum.models import RUMModel
from gps.layer.scaled_range_former_layer import ScaledRangeFormerAttention


class MotifExtractor(nn.Module):
    """Learnable motif prototype bank with soft assignment.

    Produces a soft clustering of nodes into K motif prototypes,
    enabling interpretable motif discovery from RUM path encodings.
    """

    def __init__(self, dim, num_prototypes=4, temperature=1.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.temperature = temperature

    def forward(self, h_rum):
        """
        Args:
            h_rum: [N, D] node features from RUM path encoding.

        Returns:
            motif_score: [N, K] soft assignment weights.
            motif_emb:   [N, D] weighted prototype embedding.
        """
        logits = torch.matmul(h_rum, self.prototypes.t()) / math.sqrt(h_rum.size(-1))
        motif_score = F.softmax(logits / self.temperature, dim=-1)
        motif_emb = torch.matmul(motif_score, self.prototypes)
        return motif_score, motif_emb


class SRFxRUM_MANI_Layer(nn.Module):
    """SRF x RUM — Motif-Adaptive Node Injection layer.

    Architecture (per layer):
        1. RUM forward  → h_rum
        2. Motif discovery → motif_score, motif_emb
        3. Node injection  → h_enhanced = h + α·proj(motif_emb)
        4. SRF attention   → h_srf  (zero modification to SRF core)
        5. Motif gate      → h_fused = gate·h_srf + (1-gate)·h_rum
        6. FFN + residual  → h_out
    """

    def __init__(
        self,
        dim,
        num_heads,
        srf_cfg,
        rum_cfg,
        num_prototypes=4,
        motif_temperature=1.0,
        alpha=1.0,
        dropout=0.0,
        act="relu",
        layer_norm=False,
        batch_norm=True,
        residual=True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # ------------------------------------------------------------------
        # SRF attention (zero-modification reuse)
        # ------------------------------------------------------------------
        formulation = str(getattr(srf_cfg, "formulation", "B"))
        self.srf_attn = ScaledRangeFormerAttention(
            dim_h=dim,
            num_heads=num_heads,
            attn_dropout=float(getattr(srf_cfg, "attn_dropout", 0.0)),
            formulation=formulation,
            cfg=srf_cfg,
        )

        # ------------------------------------------------------------------
        # RUM encoder (zero-modification reuse)
        # ------------------------------------------------------------------
        self.rum_use_edge_features = bool(rum_cfg.get("use_edge_features", False))
        self.rum = RUMModel(
            in_features=dim,
            out_features=dim,
            hidden_features=dim,
            depth=int(rum_cfg.get("depth", 1)),
            num_samples=int(rum_cfg.get("num_samples", 4)),
            length=int(rum_cfg.get("length", 3)),
            dropout=float(rum_cfg.get("dropout", 0.0)),
            edge_features=dim if bool(rum_cfg.get("use_edge_features", False)) else 0,
            output_softmax=False,  # keep raw features for motif discovery
            self_supervise=bool(rum_cfg.get("self_supervise", False)),
            binary=bool(rum_cfg.get("binary", False)),
        )

        # ------------------------------------------------------------------
        # Motif modules
        # ------------------------------------------------------------------
        self.motif_extractor = MotifExtractor(dim, num_prototypes, motif_temperature)
        self.motif_inject_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.fusion_gate = nn.Linear(num_prototypes, dim)
        self.alpha = alpha

        # ------------------------------------------------------------------
        # SRF output projections
        # ------------------------------------------------------------------
        self.O_h = nn.Linear(dim, dim)
        self.O_e = nn.Linear(dim, dim)

        # ------------------------------------------------------------------
        # Normalisation
        # ------------------------------------------------------------------
        if self.layer_norm:
            self.norm1_h = nn.LayerNorm(dim)
            self.norm1_e = nn.LayerNorm(dim)
            self.norm2_h = nn.LayerNorm(dim)
        if self.batch_norm:
            self.bn1_h = nn.BatchNorm1d(dim)
            self.bn1_e = nn.BatchNorm1d(dim)
            self.bn2_h = nn.BatchNorm1d(dim)

        # ------------------------------------------------------------------
        # FFN
        # ------------------------------------------------------------------
        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU}.get(act, nn.ReLU)
        self.ffn1 = nn.Linear(dim, dim * 2)
        self.act = act_cls()
        self.ffn2 = nn.Linear(dim * 2, dim)

    def _norm_h(self, h):
        if self.layer_norm:
            h = self.norm1_h(h)
        if self.batch_norm:
            h = self.bn1_h(h)
        return h

    def _norm_e(self, e):
        if self.layer_norm:
            e = self.norm1_e(e)
        if self.batch_norm:
            e = self.bn1_e(e)
        return e

    def _norm2_h(self, h):
        if self.layer_norm:
            h = self.norm2_h(h)
        if self.batch_norm:
            h = self.bn2_h(h)
        return h

    def forward(self, batch):
        """
        Args:
            batch: PyG Batch with x, edge_index, edge_attr, pair_index, pair_attr, ...

        Returns:
            batch:          updated batch (x overwritten)
            motif_score:    [N, K] soft motif assignment for interpretability
        """
        h_in = batch.x
        e_in = batch.get("pair_attr", None)

        # ------------------------------------------------------------------
        # 1. RUM pathway (unchanged)
        # ------------------------------------------------------------------
        edge_features = None
        if self.rum_use_edge_features:
            if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
                edge_features = batch.edge_attr

        h_rum, _ = self.rum(batch, h_in, e=edge_features)
        # RUMModel returns walk-level features [W, N, D]; average over walks.
        if h_rum.dim() == 3:
            h_rum = h_rum.mean(dim=0)

        # ------------------------------------------------------------------
        # 2. Motif discovery
        # ------------------------------------------------------------------
        motif_score, motif_emb = self.motif_extractor(h_rum)

        # ------------------------------------------------------------------
        # 3. Node injection  →  h_enhanced = h + α·proj(motif_emb)
        # ------------------------------------------------------------------
        h_enhanced = h_in + self.alpha * self.motif_inject_proj(motif_emb)

        # ------------------------------------------------------------------
        # 4. SRF attention (zero modification)
        #    Write h_enhanced into batch.x temporarily; SRF only reads batch.x.
        # ------------------------------------------------------------------
        batch.x = h_enhanced
        h_attn_out, e_attn_out = self.srf_attn(batch)
        batch.x = h_in  # restore original node features for safety

        h_srf = self.O_h(F.dropout(h_attn_out, p=self.dropout, training=self.training))
        e_srf = self.O_e(F.dropout(e_attn_out, p=self.dropout, training=self.training))

        # SRF residual + norm
        if self.residual:
            h_srf = h_enhanced + h_srf
            if e_in is not None:
                e_srf = e_in + e_srf

        h_srf = self._norm_h(h_srf)
        if e_in is not None:
            e_srf = self._norm_e(e_srf)

        # ------------------------------------------------------------------
        # 5. Motif-conditioned gate fusion
        # ------------------------------------------------------------------
        gate = torch.sigmoid(self.fusion_gate(motif_score))  # [N, D]
        h_fused = gate * h_srf + (1.0 - gate) * h_rum
        e_fused = e_srf  # RUM has no edge output

        # ------------------------------------------------------------------
        # 6. FFN + residual
        # ------------------------------------------------------------------
        h_ffn = self.ffn1(h_fused)
        h_ffn = self.act(h_ffn)
        h_ffn = F.dropout(h_ffn, p=self.dropout, training=self.training)
        h_ffn = self.ffn2(h_ffn)

        if self.residual:
            h_out = h_fused + h_ffn
        else:
            h_out = h_ffn

        h_out = self._norm2_h(h_out)

        # ------------------------------------------------------------------
        # Update batch state
        # ------------------------------------------------------------------
        batch.x = h_out
        if e_in is not None:
            batch.pair_attr = e_fused

        return batch, motif_score
