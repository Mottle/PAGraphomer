from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("cfg_otformer")
def set_cfg_otformer(cfg):
    """Configuration for OTFormer model and pretraining."""
    cfg.otformer = CN()

    cfg.otformer.layers = 4
    cfg.otformer.num_heads = 4
    cfg.otformer.dropout = 0.0
    cfg.otformer.attn_dropout = 0.0
    cfg.otformer.layer_norm = True
    cfg.otformer.recycling_iters = 2
    cfg.otformer.detach_recycle = True

    cfg.otformer.rum = CN()
    cfg.otformer.rum.depth = 2
    cfg.otformer.rum.num_samples = 8
    cfg.otformer.rum.rw_length = 4
    cfg.otformer.rum.dropout = 0.0
    cfg.otformer.rum.binary = False

    cfg.otformer.motif = CN()
    cfg.otformer.motif.memory_size = 64
    cfg.otformer.motif.sinkhorn_eps = 0.1
    cfg.otformer.motif.sinkhorn_iters = 5
    cfg.otformer.motif.log_domain = True

    cfg.otformer.pair = CN()
    cfg.otformer.pair.use_triangle = True
    cfg.otformer.pair.triangle_hidden = 32

    cfg.otformer.pretrain = CN()
    cfg.otformer.pretrain.enable = False
    cfg.otformer.pretrain.eval_splits = False
    cfg.otformer.pretrain.save_epoch_weights = True
    cfg.otformer.pretrain.keep_last_epoch_weights = 3
    cfg.otformer.pretrain.mode = (
        "joint"  # joint | atom_only | motif_only | edge_only | no_ot
    )
    cfg.otformer.pretrain.atom_mask_ratio = 0.15
    cfg.otformer.pretrain.motif_mask_ratio = 0.15
    cfg.otformer.pretrain.motif_topk = 8
    cfg.otformer.pretrain.edge_perturb_ratio = 0.1
    cfg.otformer.pretrain.edge_sample_ratio = 0.1
    cfg.otformer.pretrain.edge_neg_ratio = 1.0
    cfg.otformer.pretrain.w_mask_atom = 1.0
    cfg.otformer.pretrain.w_motif_mask = 1.0
    cfg.otformer.pretrain.w_edge_denoise = 1.0
    cfg.otformer.pretrain.w_ot_prior = 0.1
