from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("cfg_pag")
def set_cfg_pag(cfg):
    """Configuration for PAG model."""
    cfg.pag = CN()

    # Number of PAG layers in the network.
    cfg.pag.layers = 3

    # Default PAGLayer config shared by all layers.
    cfg.pag.layer_defaults = CN()

    cfg.pag.layer_defaults.macro = CN()
    cfg.pag.layer_defaults.macro.num_layers = 1
    cfg.pag.layer_defaults.macro.layer_type = "None+Transformer"
    cfg.pag.layer_defaults.macro.num_heads = 4
    cfg.pag.layer_defaults.macro.act = "relu"
    cfg.pag.layer_defaults.macro.dropout = 0.0
    cfg.pag.layer_defaults.macro.attn_dropout = 0.0
    cfg.pag.layer_defaults.macro.layer_norm = False
    cfg.pag.layer_defaults.macro.batch_norm = True

    cfg.pag.layer_defaults.local = CN()
    cfg.pag.layer_defaults.local.depth = 2
    cfg.pag.layer_defaults.local.num_samples = 16
    cfg.pag.layer_defaults.local.rw_length = 4
    cfg.pag.layer_defaults.local.dropout = 0.0
    cfg.pag.layer_defaults.local.binary = False

    cfg.pag.layer_defaults.path_attention = CN()
    cfg.pag.layer_defaults.path_attention.dropout = 0.0
    cfg.pag.layer_defaults.path_attention.temperature = 1.0
    cfg.pag.layer_defaults.path_attention.lambda_entropy = 0.0

    cfg.pag.layer_defaults.fusion = CN()
    cfg.pag.layer_defaults.fusion.global_fuser = "AFF"
    cfg.pag.layer_defaults.fusion.node_fuser = "AFF"

    # Optional per-layer overrides:
    # - index: target layer index (0-based)
    # - macro/local/path/fusion: partial override dict
    cfg.pag.layer_overrides = []
