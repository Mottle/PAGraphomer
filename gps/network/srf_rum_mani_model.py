import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Pretraining heads
        # ------------------------------------------------------------------
        self.mask_token = None
        self.mask_decoder = None
        self.atom_context_decoder = None
        self.mol_property_decoder = None
        self.pretrain_cfg = getattr(cfg.srf_rum_mani, "pretrain", None)
        if self.pretrain_cfg and getattr(self.pretrain_cfg, "enable", False):
            num_atom_types = int(getattr(cfg.dataset, "node_encoder_num_types", 0))
            if num_atom_types < 1:
                raise ValueError(
                    "SRFxRUM_MANI pretraining requires dataset.node_encoder=True "
                    "and dataset.node_encoder_num_types > 0."
                )
            self.mask_token = nn.Parameter(torch.zeros(1, cfg.gnn.dim_inner))
            self.mask_decoder = nn.Linear(cfg.gnn.dim_inner, num_atom_types)

            # Phase 2: atom context decoder (node-level)
            ac_cfg = getattr(self.pretrain_cfg, "atom_context", None)
            if ac_cfg and getattr(ac_cfg, "enable", False):
                ac_dim = int(getattr(ac_cfg, "dim", 158))
                self.atom_context_decoder = nn.Linear(cfg.gnn.dim_inner, ac_dim)

            # Phase 2: molecular property decoder (graph-level)
            mp_cfg = getattr(self.pretrain_cfg, "mol_property", None)
            if mp_cfg and getattr(mp_cfg, "enable", False):
                mp_num = int(getattr(mp_cfg, "num_props", 21))
                self.mol_property_decoder = nn.Sequential(
                    nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                    nn.ReLU(),
                    nn.Dropout(float(getattr(mani_cfg, "dropout", cfg.gt.dropout))),
                    nn.Linear(cfg.gnn.dim_inner, mp_num),
                )

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

    @staticmethod
    def _motif_contrastive_loss(motif_scores, node_repr, temperature=0.5):
        """Prototype dispersion loss: different prototypes should be far apart."""
        if not motif_scores:
            return torch.tensor(0.0, device=node_repr.device)
        motif_score = motif_scores[-1]  # [N, K]
        K = motif_score.size(1)
        proto_repr = torch.mm(motif_score.t(), node_repr) / (
            motif_score.sum(dim=0, keepdim=True).t() + 1e-8
        )  # [K, D]
        proto_sim = torch.mm(proto_repr, proto_repr.t()) / temperature  # [K, K]
        identity = torch.eye(K, device=proto_sim.device)
        loss = F.mse_loss(proto_sim * (1 - identity), torch.zeros_like(proto_sim))
        return loss

    @staticmethod
    def _tanimoto_similarity_matrix(fps):
        """Compute Tanimoto similarity matrix from binary fingerprints.

        fps: [B, D] float tensor (binary or continuous)
        Returns: [B, B] similarity matrix
        """
        # Jaccard / Tanimoto for binary vectors:
        # sim = (a & b).sum() / (a | b).sum()
        # Equivalent to: dot / (|a| + |b| - dot)
        fps_bool = (fps > 0).float()
        dot = torch.mm(fps_bool, fps_bool.t())  # [B, B]
        norms = fps_bool.sum(dim=1, keepdim=True)  # [B, 1]
        union = norms + norms.t() - dot
        union = torch.clamp(union, min=1e-8)
        sim = dot / union
        return sim

    @staticmethod
    def _adaptive_margin_triplet_loss(
        graph_repr, sim_matrix, margin=0.8, adaptive_coeff=1.0, dist_metric="l2norm"
    ):
        """Fingerprint-guided adaptive margin triplet loss (vectorized).

        Args:
            graph_repr: [B, D] normalized graph representations
            sim_matrix: [B, B] Tanimoto similarity matrix
            margin: base margin value
            adaptive_coeff: weight for adaptive margin component
            dist_metric: 'l2norm' or 'cossim'

        Returns:
            Combined margin + adaptive triplet loss
        """
        B = graph_repr.size(0)
        if B < 2:
            return torch.tensor(0.0, device=graph_repr.device)

        if dist_metric == "cossim":
            dist = 1 - F.cosine_similarity(
                graph_repr.unsqueeze(1), graph_repr.unsqueeze(0), dim=-1
            )
        else:
            dist = (graph_repr.unsqueeze(1) - graph_repr.unsqueeze(0)).norm(dim=-1)

        # Vectorized margin triplet: all (i,j,k) triplets at once.
        # pos_dist: dist[i,j] → [B, B, 1]
        # neg_dist: dist[i,k] → [B, 1, B]
        pos_dist = dist.unsqueeze(2)  # [B, B, 1]
        neg_dist = dist.unsqueeze(1)  # [B, 1, B]

        sim_i_j = sim_matrix.unsqueeze(2)  # [B, B, 1]
        sim_i_k = sim_matrix.unsqueeze(1)  # [B, 1, B]
        sim_diff = sim_i_j - sim_i_k  # [B, B, B]
        adaptive_margin = margin + sim_diff * margin  # [B, B, B]

        # Exclude self-pairs (i==j or i==k)
        eye = torch.eye(B, device=dist.device, dtype=torch.bool)
        valid = ~eye.unsqueeze(2) & ~eye.unsqueeze(1)  # [B, B, B], i!=j and i!=k

        loss_per_triplet = torch.clamp(
            adaptive_margin + pos_dist - neg_dist, min=0.0
        )  # [B, B, B]
        n_valid = valid.sum().float().clamp(min=1.0)
        loss_margin = loss_per_triplet[valid].sum() / n_valid

        # Hard negative mining: triplets where sim_diff > 0.3
        hard_mask = sim_diff > 0.3
        if hard_mask.any():
            loss_adaptive_per_triplet = torch.clamp(
                sim_diff.detach() * margin + pos_dist - neg_dist, min=0.0
            )
            hard_valid = hard_mask & valid
            n_hard = hard_valid.sum().float().clamp(min=1.0)
            loss_adaptive = loss_adaptive_per_triplet[hard_valid].sum() / n_hard
        else:
            loss_adaptive = torch.tensor(0.0, device=dist.device)

        return loss_margin + adaptive_coeff * loss_adaptive

    def _forward_layers(self, batch, atom_mask, atom_target):
        """Run MANI layers + decoder on an already-encoded batch."""
        all_motif_scores = []
        for layer in self.layers:
            batch, motif_score = layer(batch)
            all_motif_scores.append(motif_score)

        atom_logits = self.mask_decoder(batch.x)
        pred = atom_logits[atom_mask]
        true = atom_target[atom_mask]

        losses = {}
        losses["mask_atom"] = F.cross_entropy(pred, true)

        pt_cfg = getattr(self.pretrain_cfg, "motif_contrastive", None)
        if pt_cfg and getattr(pt_cfg, "enable", False):
            weight = float(getattr(pt_cfg, "weight", 0.5))
            temp = float(getattr(pt_cfg, "temperature", 0.5))
            losses["motif_contrastive"] = weight * self._motif_contrastive_loss(
                all_motif_scores, batch.x, temperature=temp
            )

        # Phase 2: atom context prediction
        ac_cfg = getattr(self.pretrain_cfg, "atom_context", None)
        if (
            ac_cfg
            and getattr(ac_cfg, "enable", False)
            and hasattr(batch, "atom_context")
        ):
            ac_pred = self.atom_context_decoder(batch.x)
            ac_target = batch.atom_context
            if ac_target.dim() == 1:
                ac_target = ac_target.unsqueeze(-1)
            ac_weight = float(getattr(ac_cfg, "weight", 1.0))
            losses["atom_context"] = ac_weight * F.mse_loss(ac_pred, ac_target)

        # Phase 2: molecular property prediction
        mp_cfg = getattr(self.pretrain_cfg, "mol_property", None)
        if (
            mp_cfg
            and getattr(mp_cfg, "enable", False)
            and hasattr(batch, "mol_property")
        ):
            # Graph-level pooling
            from torch_geometric.nn import global_mean_pool

            h_graph = global_mean_pool(batch.x, batch.batch)
            mp_pred = self.mol_property_decoder(h_graph)
            mp_target = batch.mol_property
            # PyG collates graph-level [21] tensors into 1D [num_graphs*21].
            num_graphs = int(batch.batch.max().item()) + 1
            mp_target = mp_target.view(num_graphs, -1)
            mp_weight = float(getattr(mp_cfg, "weight", 0.5))
            losses["mol_property"] = mp_weight * F.mse_loss(mp_pred, mp_target)

        return pred, true, losses, batch.x, all_motif_scores

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

        # Encode once
        batch = self.encoder(batch)
        batch.x = self._apply_mask_token(batch.x, atom_mask, self.mask_token)
        if hasattr(self, "pre_mp"):
            batch = self.pre_mp(batch)

        # View 1: standard
        pred, true, losses, h_out1, motif_scores1 = self._forward_layers(
            batch, atom_mask, atom_target
        )

        # ---- MolMCL-style graph-level contrastive learning ----
        from torch_geometric.nn import global_mean_pool

        h_graph = global_mean_pool(h_out1, batch.batch)  # [num_graphs, D]

        # Molecular fingerprint-guided contrastive learning
        fp_cfg = getattr(self.pretrain_cfg, "fingerprint_contrastive", None)
        if fp_cfg and getattr(fp_cfg, "enable", False) and hasattr(batch, "mol_fp"):
            fp_weight = float(getattr(fp_cfg, "weight", 1.0))
            fp_margin = float(getattr(fp_cfg, "margin", 0.8))
            fp_adaptive = float(getattr(fp_cfg, "adaptive_coeff", 1.0))

            h_graph_norm = F.normalize(h_graph, dim=-1)
            mol_fps = batch.mol_fp
            num_graphs = h_graph.size(0)
            fp_dim = mol_fps.numel() // num_graphs
            mol_fps = mol_fps.view(num_graphs, fp_dim)
            sim_matrix = self._tanimoto_similarity_matrix(mol_fps)
            loss_mol = self._adaptive_margin_triplet_loss(
                h_graph_norm,
                sim_matrix,
                margin=fp_margin,
                adaptive_coeff=fp_adaptive,
            )
            losses["fingerprint_contrastive"] = fp_weight * loss_mol

        # Scaffold-level contrastive learning
        sc_cfg = getattr(self.pretrain_cfg, "scaffold_contrastive", None)
        if sc_cfg and getattr(sc_cfg, "enable", False) and hasattr(batch, "scaff_fp"):
            sc_weight = float(getattr(sc_cfg, "weight", 1.0))
            sc_margin = float(getattr(sc_cfg, "margin", 0.8))
            sc_adaptive = float(getattr(sc_cfg, "adaptive_coeff", 1.0))

            h_graph_norm = F.normalize(h_graph, dim=-1)
            scaff_fps = batch.scaff_fp
            num_graphs = h_graph.size(0)
            fp_dim = scaff_fps.numel() // num_graphs
            scaff_fps = scaff_fps.view(num_graphs, fp_dim)
            scaff_sim_matrix = self._tanimoto_similarity_matrix(scaff_fps)
            loss_scaff = self._adaptive_margin_triplet_loss(
                h_graph_norm,
                scaff_sim_matrix,
                margin=sc_margin,
                adaptive_coeff=sc_adaptive,
            )
            losses["scaffold_contrastive"] = sc_weight * loss_scaff

        # ---- Perturbation contrastive learning (Gap 1: same-molecule pairs) ----
        perturb_cfg = getattr(self.pretrain_cfg, "perturbation_contrastive", None)
        if (
            perturb_cfg
            and getattr(perturb_cfg, "enable", False)
            and hasattr(batch, "_perturb_data")
        ):
            perturb_weight = float(getattr(perturb_cfg, "weight", 1.0))
            from torch_geometric.data import Batch as PyGBatch

            pdata_list = batch._perturb_data  # list of Data or None
            perturb_graphs = []
            valid = []

            for pdata in pdata_list:
                if pdata is not None:
                    perturb_graphs.append(pdata)
                    valid.append(True)
                else:
                    perturb_graphs.append(
                        type(batch)(
                            x=torch.zeros(1, 1, dtype=torch.long),
                            edge_index=torch.zeros(2, 0, dtype=torch.long),
                        )
                    )
                    valid.append(False)

            valid_mask = torch.tensor(valid, device=batch.x.device, dtype=torch.bool)
            if valid_mask.any():
                perturb_batch = PyGBatch.from_data_list(perturb_graphs).to(
                    batch.x.device
                )
                perturb_enc = self.encoder(perturb_batch)
                h_perturb = global_mean_pool(perturb_enc.x, perturb_batch.batch)
                h_orig = global_mean_pool(h_out1, batch.batch)

                h_o = F.normalize(h_orig[valid_mask], dim=-1)
                h_p = F.normalize(h_perturb[valid_mask], dim=-1)
                cos = (h_o * h_p).sum(dim=-1)
                losses["perturbation_contrastive"] = perturb_weight * (1.0 - cos.mean())

        # View 2: multi-view dropout contrastive
        vd_cfg = getattr(self.pretrain_cfg, "view_dropout", None)
        if vd_cfg and getattr(vd_cfg, "enable", False):
            vd_ratio = float(getattr(vd_cfg, "ratio", 0.3))
            weight = float(getattr(vd_cfg, "weight", 0.3))

            # Clone encoded batch and perturb pair_attr (zero-out, keep index)
            batch_v2 = type(batch)(
                x=batch.x.clone(),
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
            )
            for k, v in batch:
                if k not in ("x", "edge_index", "edge_attr", "batch"):
                    setattr(batch_v2, k, v.clone() if hasattr(v, "clone") else v)

            if hasattr(batch_v2, "pair_attr") and batch_v2.pair_attr is not None:
                num_pairs = batch_v2.pair_attr.size(0)
                drop_mask = (
                    torch.rand(num_pairs, device=batch_v2.pair_attr.device) < vd_ratio
                )
                if drop_mask.any():
                    batch_v2.pair_attr = batch_v2.pair_attr.clone()
                    batch_v2.pair_attr[drop_mask] = 0.0

            _, _, _, h_out2, _ = self._forward_layers(batch_v2, atom_mask, atom_target)

            # Node-level consistency loss
            h1_norm = F.normalize(h_out1, dim=-1)
            h2_norm = F.normalize(h_out2, dim=-1)
            sim = (h1_norm * h2_norm).sum(dim=-1)
            losses["view_consistency"] = weight * (1.0 - sim.mean())

        batch.mani_aux_losses = losses
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

        batch.mani_motif_scores = all_motif_scores
        return self.post_mp(batch)
