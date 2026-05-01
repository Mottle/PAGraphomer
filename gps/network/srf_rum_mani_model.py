import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network

from gps.layer.srf_rum_mani_layer import SRFxRUM_MANI_Layer
from gps.network.grit_model import FeatureEncoder


@register_network("SRFxRUM_MANI_Model")
class SRFxRUM_MANI_Model(nn.Module):
    """SRF x RUM — Motif-Adaptive Node Injection model.

    Stacks SRFxRUM_MANI_Layer layers after the shared FeatureEncoder.
    Each layer performs dual-pathway (SRF + RUM) forward with motif
    discovery, node-level injection, and motif-conditioned gate fusion.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if cfg.gnn.dim_inner != dim_in:
            raise ValueError(
                f"The inner and hidden dims must match for SRFxRUM_MANI: "
                f"gnn.dim_inner={cfg.gnn.dim_inner}, dim_in={dim_in}"
            )

        mani_cfg = getattr(cfg, "srf_rum_mani", None)
        if mani_cfg is None:
            raise ValueError(
                "SRFxRUM_MANI_Model requires cfg.srf_rum_mani section. "
                "Ensure gps.config.srf_rum_mani_config is loaded."
            )

        # SRF sub-config: use cfg.gt for msrrwp/thresholds/attn settings.
        # mani_cfg.srf only carries overrides (formulation, attn_dropout).
        srf_cfg = cfg.gt
        rum_cfg = getattr(mani_cfg, "rum", {})

        layers = []
        for _ in range(getattr(mani_cfg, "layers", cfg.gt.layers)):
            layers.append(
                SRFxRUM_MANI_Layer(
                    dim=cfg.gnn.dim_inner,
                    num_heads=cfg.gt.n_heads,
                    srf_cfg=srf_cfg,
                    rum_cfg=rum_cfg,
                    num_prototypes=int(getattr(mani_cfg, "num_prototypes", 4)),
                    motif_temperature=float(
                        getattr(mani_cfg, "motif_temperature", 1.0)
                    ),
                    alpha=float(getattr(mani_cfg, "alpha", 1.0)),
                    dropout=float(getattr(mani_cfg, "dropout", cfg.gt.dropout)),
                    act=cfg.gnn.act,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    residual=True,
                )
            )
        self.layers = nn.ModuleList(layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        # ------------------------------------------------------------------
        # Pretraining head (masked node feature reconstruction)
        # ------------------------------------------------------------------
        self.mask_token = None
        self.mask_decoder = None
        if getattr(cfg.srf_rum_mani.pretrain, "enable", False):
            num_atom_types = int(getattr(cfg.dataset, "node_encoder_num_types", 0))
            if num_atom_types < 1:
                raise ValueError(
                    "SRFxRUM_MANI pretraining requires dataset.node_encoder=True "
                    "and dataset.node_encoder_num_types > 0."
                )
            self.mask_token = nn.Parameter(torch.zeros(1, cfg.gnn.dim_inner))
            self.mask_decoder = nn.Linear(cfg.gnn.dim_inner, num_atom_types)

    @staticmethod
    def _sample_pretrain_mask(num_nodes, device, ratio=0.15):
        if ratio <= 0.0 or num_nodes == 0:
            return torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask = torch.rand(num_nodes, device=device) < ratio
        if not mask.any():
            mask[torch.randint(num_nodes, (1,), device=device)] = True
        return mask

    @staticmethod
    def _apply_mask_token(x, mask, mask_token):
        if not mask.any():
            return x
        x = x.clone()
        x[mask] = mask_token.expand(int(mask.sum().item()), -1)
        return x

    def forward_pretrain(self, batch):
        if not cfg.dataset.node_encoder:
            raise RuntimeError(
                "SRFxRUM_MANI masked pretraining requires dataset.node_encoder=True."
            )
        if batch.x.dim() < 2:
            raise RuntimeError(
                "SRFxRUM_MANI masked pretraining expects raw categorical "
                "node features in batch.x."
            )

        atom_target = batch.x[:, 0].long()
        ratio = float(getattr(cfg.srf_rum_mani.pretrain, "mask_ratio", 0.15))
        atom_mask = self._sample_pretrain_mask(
            atom_target.size(0), atom_target.device, ratio
        )

        batch = self.encoder(batch)
        batch.x = self._apply_mask_token(batch.x, atom_mask, self.mask_token)

        if hasattr(self, "pre_mp"):
            batch = self.pre_mp(batch)

        for layer in self.layers:
            batch, _ = layer(batch)

        atom_logits = self.mask_decoder(batch.x)
        pred = atom_logits[atom_mask]
        true = atom_target[atom_mask]

        batch.mani_motif_scores = getattr(batch, "mani_motif_scores", [])
        return pred, true

    def forward(self, batch):
        if getattr(cfg.srf_rum_mani.pretrain, "enable", False):
            return self.forward_pretrain(batch)

        batch = self.encoder(batch)
        if hasattr(self, "pre_mp"):
            batch = self.pre_mp(batch)

        all_motif_scores = []

        for layer in self.layers:
            batch, motif_score = layer(batch)
            all_motif_scores.append(motif_score)

        # Attach auxiliary artefacts for downstream logging / interpretability
        batch.mani_motif_scores = all_motif_scores

        return self.post_mp(batch)
