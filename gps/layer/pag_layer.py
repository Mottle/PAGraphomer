import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from gps.layer.gps_layer import GPSLayer
from gps.layer.pag.fusion import AFF, IAFF
from gps.layer.pag.path_attention import PathAttention
from gps.layer.rum.models import RUMModel


class PAGLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        pooler=global_mean_pool,
        macro_cfg=None,
        local_cfg=None,
        path_cfg=None,
        fusion_cfg=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        macro_cfg = {} if macro_cfg is None else macro_cfg
        local_cfg = {} if local_cfg is None else local_cfg
        path_cfg = {} if path_cfg is None else path_cfg
        fusion_cfg = {} if fusion_cfg is None else fusion_cfg

        num_me_layers = int(macro_cfg.get("num_layers", 1))
        macro_gps_layer_type = macro_cfg.get("layer_type", "None+Transformer")
        macro_gps_num_heads = int(macro_cfg.get("num_heads", 4))
        macro_gps_act = macro_cfg.get("act", "relu")
        macro_dropout = float(macro_cfg.get("dropout", 0.0))
        macro_gps_attn_dropout = float(macro_cfg.get("attn_dropout", 0.0))
        macro_gps_layer_norm = bool(macro_cfg.get("layer_norm", False))
        macro_gps_batch_norm = bool(macro_cfg.get("batch_norm", True))
        macro_gps_pna_degrees = macro_cfg.get("pna_degrees", None)
        macro_gps_equivstable_pe = bool(macro_cfg.get("equivstable_pe", False))
        macro_gps_bigbird_cfg = macro_cfg.get("bigbird_cfg", None)
        macro_gps_log_attn_weights = bool(macro_cfg.get("log_attn_weights", False))

        local_depth = int(local_cfg.get("depth", 2))
        local_num_samples = int(local_cfg.get("num_samples", 16))
        local_rw_length = int(local_cfg.get("rw_length", 4))
        local_dropout = float(local_cfg.get("dropout", 0.0))
        local_binary = bool(local_cfg.get("binary", False))

        path_dropout = float(path_cfg.get("dropout", 0.0))
        path_temp = float(path_cfg.get("temperature", path_cfg.get("temp", 1.0)))
        path_lambda_entropy = float(path_cfg.get("lambda_entropy", 0.0))

        global_fuser = fusion_cfg.get("global_fuser", AFF)
        node_fuser = fusion_cfg.get("node_fuser", AFF)

        if num_me_layers < 1:
            raise ValueError(f"`macro.num_layers` must be >= 1, got {num_me_layers}.")
        try:
            local_gnn_type, global_model_type = macro_gps_layer_type.split("+")
        except ValueError as exc:
            raise ValueError(
                "Unexpected `macro.layer_type`: "
                f"{macro_gps_layer_type}. Expected format '<local>+<global>'."
            ) from exc

        macro_layers = []
        for _ in range(num_me_layers):
            macro_layers.append(
                GPSLayer(
                    dim_h=channels,
                    local_gnn_type=local_gnn_type,
                    global_model_type=global_model_type,
                    num_heads=macro_gps_num_heads,
                    act=macro_gps_act,
                    pna_degrees=macro_gps_pna_degrees,
                    equivstable_pe=macro_gps_equivstable_pe,
                    dropout=macro_dropout,
                    attn_dropout=macro_gps_attn_dropout,
                    layer_norm=macro_gps_layer_norm,
                    batch_norm=macro_gps_batch_norm,
                    bigbird_cfg=macro_gps_bigbird_cfg,
                    log_attn_weights=macro_gps_log_attn_weights,
                )
            )
        self.macro_encoder = nn.Sequential(*macro_layers)

        self.local_encoder = RUMModel(
            in_features=channels,
            out_features=channels,
            hidden_features=channels,
            edge_features=channels,
            depth=local_depth,
            num_samples=local_num_samples,
            length=local_rw_length,
            dropout=local_dropout,
            binary=local_binary,
        )

        self.path_attention = PathAttention(
            channels,
            dropout=path_dropout,
            temp=path_temp,
            lambda_entropy=path_lambda_entropy,
        )
        self.pooler = pooler
        self.global_fuser = global_fuser(channels)
        self.node_fuser = node_fuser(channels)

    def forward_local(self, h, data):
        out, ss_loss = self.local_encoder(data, h, data.edge_attr)
        return out, ss_loss

    def forward_macro(self, h, data):
        batch = data.clone()
        batch.x = h
        batch = self.macro_encoder(batch)
        return batch.x, 0.0

    def readout(self, h, data):
        return self.pooler(h, data.batch)

    def forward(self, data):
        h = data.x
        struct_features, ss_loss = self.forward_local(h, data)
        macro_features, _ = self.forward_macro(h, data)
        global_feature = self.readout(macro_features, data)

        attn_struct_feature, attn_weights, attn_entropy_loss = self.path_attention(
            global_feature, struct_features, data.batch
        )

        fused_global_feature = self.global_fuser(global_feature, attn_struct_feature)
        struct_features = struct_features.mean(dim=0)
        fused_h = self.node_fuser(macro_features, struct_features)

        loss = attn_entropy_loss + ss_loss

        return fused_h, fused_global_feature, attn_weights, loss
