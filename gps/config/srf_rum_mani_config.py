from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("srf_rum_mani_config")
def srf_rum_mani_cfg(cfg):
    """Configuration schema for SRFxRUM_MANI model."""

    if not hasattr(cfg, "srf_rum_mani"):
        cfg.srf_rum_mani = CN()

    # Layer stack
    cfg.srf_rum_mani.layers = 6
    cfg.srf_rum_mani.dim_hidden = 32

    # Motif prototype bank
    cfg.srf_rum_mani.num_prototypes = 4
    cfg.srf_rum_mani.motif_temperature = 1.0

    # Node injection strength (α in h + α·proj(motif_emb))
    cfg.srf_rum_mani.alpha = 1.0

    # Dropout for the MANI layer internals (FFN, projections)
    cfg.srf_rum_mani.dropout = 0.0

    # Pretraining
    cfg.srf_rum_mani.pretrain = CN()
    cfg.srf_rum_mani.pretrain.enable = False
    cfg.srf_rum_mani.pretrain.mask_ratio = 0.15

    # Motif contrastive learning (prototype dispersion)
    cfg.srf_rum_mani.pretrain.motif_contrastive = CN()
    cfg.srf_rum_mani.pretrain.motif_contrastive.enable = True
    cfg.srf_rum_mani.pretrain.motif_contrastive.weight = 0.5
    cfg.srf_rum_mani.pretrain.motif_contrastive.temperature = 0.5

    # Multi-view dropout contrastive (node consistency)
    cfg.srf_rum_mani.pretrain.view_dropout = CN()
    cfg.srf_rum_mani.pretrain.view_dropout.enable = True
    cfg.srf_rum_mani.pretrain.view_dropout.weight = 0.3
    cfg.srf_rum_mani.pretrain.view_dropout.ratio = 0.3

    # Atom context prediction (Phase 2)
    cfg.srf_rum_mani.pretrain.atom_context = CN()
    cfg.srf_rum_mani.pretrain.atom_context.enable = True
    cfg.srf_rum_mani.pretrain.atom_context.weight = 1.0
    cfg.srf_rum_mani.pretrain.atom_context.dim = 158

    # Molecular property prediction (Phase 2)
    cfg.srf_rum_mani.pretrain.mol_property = CN()
    cfg.srf_rum_mani.pretrain.mol_property.enable = True
    cfg.srf_rum_mani.pretrain.mol_property.weight = 0.5
    cfg.srf_rum_mani.pretrain.mol_property.num_props = 21

    # Molecular fingerprint-guided contrastive learning (Phase 3)
    cfg.srf_rum_mani.pretrain.fingerprint_contrastive = CN()
    cfg.srf_rum_mani.pretrain.fingerprint_contrastive.enable = True
    cfg.srf_rum_mani.pretrain.fingerprint_contrastive.weight = 1.0
    cfg.srf_rum_mani.pretrain.fingerprint_contrastive.margin = 0.8
    cfg.srf_rum_mani.pretrain.fingerprint_contrastive.adaptive_coeff = 1.0
    cfg.srf_rum_mani.pretrain.fingerprint_contrastive.temperature = 0.1

    # Scaffold-level contrastive learning (Phase 3)
    cfg.srf_rum_mani.pretrain.scaffold_contrastive = CN()
    cfg.srf_rum_mani.pretrain.scaffold_contrastive.enable = True
    cfg.srf_rum_mani.pretrain.scaffold_contrastive.weight = 1.0
    cfg.srf_rum_mani.pretrain.scaffold_contrastive.margin = 0.8
    cfg.srf_rum_mani.pretrain.scaffold_contrastive.adaptive_coeff = 1.0
    cfg.srf_rum_mani.pretrain.scaffold_contrastive.temperature = 0.1

    # Scaffold-invariant perturbation contrastive learning (Gap 1)
    cfg.srf_rum_mani.pretrain.perturbation_contrastive = CN()
    cfg.srf_rum_mani.pretrain.perturbation_contrastive.enable = False
    cfg.srf_rum_mani.pretrain.perturbation_contrastive.weight = 1.0
    cfg.srf_rum_mani.pretrain.perturbation_contrastive.crem_db = (
        "datasets/chembl22_sa2.db"
    )
    cfg.srf_rum_mani.pretrain.perturbation_contrastive.max_size = 5
    cfg.srf_rum_mani.pretrain.perturbation_contrastive.radius = 2

    # SRF sub-config (forwarded to ScaledRangeFormerAttention)
    cfg.srf_rum_mani.srf = CN()
    cfg.srf_rum_mani.srf.formulation = "B"  # A | B | C | D
    cfg.srf_rum_mani.srf.attn_dropout = 0.0

    # RUM sub-config (forwarded to RUMModel)
    cfg.srf_rum_mani.rum = CN()
    cfg.srf_rum_mani.rum.num_samples = 4  # W: walks per node
    cfg.srf_rum_mani.rum.length = 3  # L: walk length
    cfg.srf_rum_mani.rum.depth = 1  # internal RUM layers
    cfg.srf_rum_mani.rum.dropout = 0.0
    cfg.srf_rum_mani.rum.self_supervise = False
    cfg.srf_rum_mani.rum.binary = False
    cfg.srf_rum_mani.rum.use_edge_features = False

    return cfg
