from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("grit_config")
def grit_cfg(cfg):
    """GRIT-specific configuration defaults."""
    cfg.grit = CN()
    cfg.grit.pretrain = CN()
    cfg.grit.pretrain.enable = False
    cfg.grit.pretrain.eval_splits = False
    cfg.grit.pretrain.save_epoch_weights = True
    cfg.grit.pretrain.keep_last_epoch_weights = 3
    cfg.grit.pretrain.mask_ratio = 0.15
    cfg.grit.pretrain.edge_mask_ratio = 0.15
    cfg.grit.pretrain.w_mask_atom = 1.0
    cfg.grit.pretrain.w_mask_edge = 1.0

    cfg.grit.motil_pretrain = CN()
    cfg.grit.motil_pretrain.enable = False
    cfg.grit.motil_pretrain.eval_splits = False
    cfg.grit.motil_pretrain.save_epoch_weights = True
    cfg.grit.motil_pretrain.keep_last_epoch_weights = 3
    cfg.grit.motil_pretrain.dropout1 = 0.3
    cfg.grit.motil_pretrain.dropout2 = 0.3
    cfg.grit.motil_pretrain.depth1 = -1
    cfg.grit.motil_pretrain.depth2 = -1
    cfg.grit.motil_pretrain.temperature = 0.1
    cfg.grit.motil_pretrain.diffusion_steps = 1000
    cfg.grit.motil_pretrain.diffusion_hidden = 64
    cfg.grit.motil_pretrain.diffusion_lr = 3e-5
    cfg.grit.motil_pretrain.contrast_lr = 3e-5
    cfg.grit.motil_pretrain.scheduler_gamma = 0.99
    cfg.grit.motil_pretrain.scheduler_step = 500
    cfg.grit.motil_pretrain.w_diffusion = 1.0
    cfg.grit.motil_pretrain.w_contrast = 1.0
    cfg.grit.motil_pretrain.w_fgs = 1.0
    cfg.grit.motil_pretrain.task_cycle = "alternate"
    cfg.grit.motil_pretrain.loaded_feature_cache = False

    cfg.grit.finetune = CN()
    cfg.grit.finetune.enable = False
    cfg.grit.finetune.freeze_backbone = False
    cfg.grit.finetune.freeze_backbone_epochs = 0
    cfg.grit.finetune.backbone_lr_ratio = 0.1
    cfg.grit.finetune.encoder_lr_ratio = 1.0
    cfg.grit.finetune.train_encoder_when_freeze_backbone = True
