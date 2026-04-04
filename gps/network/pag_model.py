import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network
from yacs.config import CfgNode as CN

from gps.layer.pag.fusion import AFF, IAFF
from gps.layer.pag_layer import PAGLayer
from gps.network.gps_model import FeatureEncoder


def _resolve_fuser(name: str):
    fuser_dict = {
        "AFF": AFF,
        "IAFF": IAFF,
    }
    if name not in fuser_dict:
        raise ValueError(
            f"Unsupported fuser '{name}'. "
            f"Supported values: {list(fuser_dict.keys())}"
        )
    return fuser_dict[name]


def _cfg_node_to_dict(obj):
    if isinstance(obj, CN):
        out = {}
        for key, value in obj.items():
            out[key] = _cfg_node_to_dict(value)
        return out
    if isinstance(obj, list):
        return [_cfg_node_to_dict(v) for v in obj]
    return obj


def _deep_update(base, updates):
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@register_network("PAGModel")
class PAGModel(torch.nn.Module):
    """PAG network with GPS-based macro encoder and RUM local encoder."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if cfg.gnn.dim_inner != dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"dim_inner={cfg.gnn.dim_inner}, dim_in={dim_in}"
            )

        pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        base_layer_cfg = _cfg_node_to_dict(cfg.pag.layer_defaults)
        base_layer_cfg["macro"]["act"] = cfg.gnn.act
        base_layer_cfg["macro"]["pna_degrees"] = cfg.gt.pna_degrees
        base_layer_cfg["macro"]["equivstable_pe"] = cfg.posenc_EquivStableLapPE.enable
        base_layer_cfg["macro"]["bigbird_cfg"] = cfg.gt.bigbird
        base_layer_cfg["macro"]["log_attn_weights"] = (
            cfg.train.mode == "log-attn-weights"
        )

        raw_overrides = _cfg_node_to_dict(cfg.pag.layer_overrides)
        override_by_index = {}
        for entry in raw_overrides:
            if "index" not in entry:
                raise ValueError(
                    "Each `pag.layer_overrides` entry must define `index`."
                )
            idx = int(entry["index"])
            if idx < 0 or idx >= cfg.pag.layers:
                raise ValueError(
                    f"`pag.layer_overrides.index` out of range: {idx}. "
                    f"Expected [0, {cfg.pag.layers - 1}]."
                )
            override_cfg = {k: v for k, v in entry.items() if k != "index"}
            if idx in override_by_index:
                _deep_update(override_by_index[idx], override_cfg)
            else:
                override_by_index[idx] = override_cfg

        layers = []
        for layer_idx in range(cfg.pag.layers):
            layer_cfg = {
                "macro": dict(base_layer_cfg["macro"]),
                "local": dict(base_layer_cfg["local"]),
                "path_attention": dict(base_layer_cfg["path_attention"]),
                "fusion": dict(base_layer_cfg["fusion"]),
            }
            if layer_idx in override_by_index:
                _deep_update(layer_cfg, override_by_index[layer_idx])
            if "path" in layer_cfg and "path_attention" not in layer_cfg:
                # Backward-compatible alias for old configs.
                layer_cfg["path_attention"] = layer_cfg["path"]

            fusion_cfg = dict(layer_cfg["fusion"])
            fusion_cfg["global_fuser"] = _resolve_fuser(
                layer_cfg["fusion"]["global_fuser"]
            )
            fusion_cfg["node_fuser"] = _resolve_fuser(layer_cfg["fusion"]["node_fuser"])

            layers.append(
                PAGLayer(
                    channels=cfg.gnn.dim_inner,
                    pooler=pooling_fun,
                    macro_cfg=layer_cfg["macro"],
                    local_cfg=layer_cfg["local"],
                    path_cfg=layer_cfg["path_attention"],
                    fusion_cfg=fusion_cfg,
                )
            )
        self.pag_layers = torch.nn.ModuleList(layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        batch = self.encoder(batch)
        if hasattr(self, "pre_mp"):
            batch = self.pre_mp(batch)

        pag_loss = 0.0
        for layer in self.pag_layers:
            x, global_feature, attn_weights, aux_loss = layer(batch)
            batch.x = x
            batch.pag_global_feature = global_feature
            batch.pag_attn_weights = attn_weights
            pag_loss = pag_loss + aux_loss
        batch.pag_loss = pag_loss

        return self.post_mp(batch)
