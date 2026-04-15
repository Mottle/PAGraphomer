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
    cfg.otformer.ffn_activation = "gelu"
    cfg.otformer.recycling_iters = 2
    cfg.otformer.detach_recycle = True

    cfg.otformer.rum = CN()
    cfg.otformer.rum.depth = 2
    cfg.otformer.rum.num_samples = 8
    cfg.otformer.rum.rw_length = 4
    cfg.otformer.rum.dropout = 0.0
    cfg.otformer.rum.binary = False
    cfg.otformer.rum.output_softmax = True

    cfg.otformer.motif = CN()
    cfg.otformer.motif.memory_size = 64
    cfg.otformer.motif.sinkhorn_eps = 0.1
    cfg.otformer.motif.sinkhorn_iters = 5
    cfg.otformer.motif.log_domain = True

    cfg.otformer.pair = CN()
    cfg.otformer.pair.use_triangle = True
    cfg.otformer.pair.triangle_hidden = 32
    # Precompute shortest-path distance (SPD) and inject into z0.
    cfg.otformer.pair.use_spd = True
    # Distances are clipped to this value during preprocessing.
    cfg.otformer.pair.spd_max_dist = 16
    # GRIT-style relative positional bias from RRWP for DTBlock attention logits.
    cfg.otformer.pair.use_rrwp = False
    cfg.otformer.pair.rrwp_attr_name = "rrwp"
    cfg.otformer.pair.rrwp_dim = 0

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
    cfg.otformer.pretrain.edge_denoise_mode = "hard_spd"
    cfg.otformer.pretrain.hard_neg_max_spd = 3
    cfg.otformer.pretrain.w_mask_atom = 1.0
    cfg.otformer.pretrain.w_motif_mask = 1.0
    cfg.otformer.pretrain.w_edge_denoise = 1.0
    cfg.otformer.pretrain.w_ot_prior = 0.1

    cfg.otformer.finetune = CN()
    cfg.otformer.finetune.enable = False
    cfg.otformer.finetune.freeze_backbone = True
    cfg.otformer.finetune.freeze_backbone_epochs = 0
    cfg.otformer.finetune.backbone_lr_ratio = 0.1
    cfg.otformer.finetune.train_encoder_when_freeze_backbone = True
    cfg.otformer.finetune.encoder_lr_ratio = 1.0
    cfg.otformer.finetune.readout_dim = 256
    cfg.otformer.finetune.pooling_z = "mean"
    cfg.otformer.finetune.cls_hidden = 128
    cfg.otformer.finetune.cls_mlp_activation = "gelu"
    cfg.otformer.finetune.cls_activation = "none"
    cfg.otformer.finetune.reg_hidden = 128
    cfg.otformer.finetune.reg_mlp_activation = "gelu"
    cfg.otformer.finetune.asl_gamma_pos = 1.0
    cfg.otformer.finetune.asl_gamma_neg = 4.0
    cfg.otformer.finetune.asl_clip = 0.05

    cfg.otformer.ablation = CN()
    cfg.otformer.ablation.enable = False
    cfg.otformer.ablation.disable_rum_ot = False
    cfg.otformer.ablation.readout_variant = "full"
