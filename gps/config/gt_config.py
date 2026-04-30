from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("cfg_gt")
def set_cfg_gt(cfg):
    """Configuration for Graph Transformer-style models, e.g.:
    - Spectral Attention Network (SAN) Graph Transformer.
    - "vanilla" Transformer / Performer.
    - General Powerful Scalable (GPS) Model.
    """

    # Positional encodings argument group
    cfg.gt = CN()

    # Type of Graph Transformer layer to use
    cfg.gt.layer_type = "SANLayer"

    # Number of Transformer layers in the model
    cfg.gt.layers = 3

    # Number of attention heads in the Graph Transformer
    cfg.gt.n_heads = 8

    # Size of the hidden node and edge representation
    cfg.gt.dim_hidden = 64

    # Full attention SAN transformer including all possible pairwise edges
    cfg.gt.full_graph = True

    # SAN real vs fake edge attention weighting coefficient
    cfg.gt.gamma = 1e-5

    # Histogram of in-degrees of nodes in the training set used by PNAConv.
    # Used when `gt.layer_type: PNAConv+...`. If empty it is precomputed during
    # the dataset loading process.
    cfg.gt.pna_degrees = []

    # Dropout in feed-forward module.
    cfg.gt.dropout = 0.0

    # Dropout in self-attention.
    cfg.gt.attn_dropout = 0.0

    cfg.gt.layer_norm = False

    cfg.gt.batch_norm = True

    cfg.gt.bn_momentum = 0.1

    cfg.gt.bn_no_runner = False

    cfg.gt.residual = True

    cfg.gt.update_e = True

    cfg.gt.rezero = False

    cfg.gt.attn = CN()
    cfg.gt.attn.use = True
    cfg.gt.attn.sparse = False
    cfg.gt.attn.deg_scaler = True
    cfg.gt.attn.use_bias = False
    cfg.gt.attn.clamp = 5.0
    cfg.gt.attn.act = "relu"
    cfg.gt.attn.full_attn = True
    cfg.gt.attn.norm_e = True
    cfg.gt.attn.O_e = True
    cfg.gt.attn.edge_enhance = True

    # RRWP multi-scale pair transformer configuration.
    cfg.gt.msrrwp = CN()
    cfg.gt.msrrwp.rrwp_name = "rrwp"
    cfg.gt.msrrwp.use_spd = True
    cfg.gt.msrrwp.spd_name = "srf_spd"
    cfg.gt.msrrwp.spd_max_dist = 16
    cfg.gt.msrrwp.thresholds = [0.3, 0.6, 1.0]
    cfg.gt.msrrwp.alphas = [1.0, 1.0, 1.0]
    cfg.gt.msrrwp.scale_mode = "percentile"
    cfg.gt.msrrwp.mask_mode = "hard"
    cfg.gt.msrrwp.steps = [1, 3, 8]
    cfg.gt.msrrwp.soft_eps = 1e-6
    cfg.gt.msrrwp.hard_eps = 1e-9
    cfg.gt.msrrwp.weight_in_softmax = False
    cfg.gt.msrrwp.learnable_weights = True
    cfg.gt.msrrwp.inner_residual = True
    cfg.gt.msrrwp.inner_norm = True
    cfg.gt.msrrwp.inject_edge_attr = True
    cfg.gt.msrrwp.add_adj_indicator = True

    # BigBird model/GPS-BigBird layer.
    cfg.gt.bigbird = CN()

    cfg.gt.bigbird.attention_type = "block_sparse"

    cfg.gt.bigbird.chunk_size_feed_forward = 0

    cfg.gt.bigbird.is_decoder = False

    cfg.gt.bigbird.add_cross_attention = False

    cfg.gt.bigbird.hidden_act = "relu"

    cfg.gt.bigbird.max_position_embeddings = 128

    cfg.gt.bigbird.use_bias = False

    cfg.gt.bigbird.num_random_blocks = 3

    cfg.gt.bigbird.block_size = 3

    cfg.gt.bigbird.layer_norm_eps = 1e-6

    # GatedDeltaNet: permutation ensemble size (P)
    cfg.gt.perm_ensemble = 1
    # GatedDeltaNet: consistency regularization weight
    cfg.gt.perm_lambda = 1.0
    cfg.gt.perm_mode = "none"
    cfg.gt.perm_pct = 20
    cfg.gt.gated_attn = False

    # GatedDeltaNet PE encoder dimensions
    cfg.gt.pe_dim_x = 16
    cfg.gt.pe_dim_rwse = 6
    cfg.gt.pe_dim_lap = 6
    cfg.gt.pe_dim_deg = 4

    # GatedDeltaNet FLA parameters
    cfg.gt.gdn_short_conv = False
    cfg.gt.gdn_head_dim = 16
    cfg.gt.gdn_expand_v = 2
    cfg.gt.dual_fla = True
