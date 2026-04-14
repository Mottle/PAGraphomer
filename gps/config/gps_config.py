from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("cfg_gps")
def set_cfg_gps(cfg):
    """Configuration for GPS pretraining and downstream finetuning."""

    cfg.gps = CN()

    cfg.gps.pretrain = CN()
    cfg.gps.pretrain.enable = False
    cfg.gps.pretrain.eval_splits = False
    cfg.gps.pretrain.save_epoch_weights = True
    cfg.gps.pretrain.keep_last_epoch_weights = 3
    cfg.gps.pretrain.mask_ratio = 0.15
    cfg.gps.pretrain.edge_mask_ratio = 0.15
    cfg.gps.pretrain.w_mask_atom = 1.0
    cfg.gps.pretrain.w_mask_edge = 1.0

    cfg.gps.finetune = CN()
    cfg.gps.finetune.enable = False
    cfg.gps.finetune.freeze_backbone = True
    cfg.gps.finetune.freeze_backbone_epochs = 0
    cfg.gps.finetune.backbone_lr_ratio = 0.1
    cfg.gps.finetune.train_encoder_when_freeze_backbone = True
    cfg.gps.finetune.encoder_lr_ratio = 1.0
