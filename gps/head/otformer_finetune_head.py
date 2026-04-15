import torch
import torch.nn as nn
from torch_geometric.graphgym import register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_head
from torch_scatter import scatter_mean, scatter_max, scatter_add


@register_head("otformer_finetune")
class OTFormerFineTuneHead(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_h = dim_in
        self.motif_size = cfg.otformer.motif.memory_size
        readout_dim = cfg.otformer.finetune.readout_dim
        self.readout_variant = getattr(cfg.otformer.ablation, "readout_variant", "full")

        if self.readout_variant == "full":
            fusion_dim = (3 * dim_h) + self.motif_size
        elif self.readout_variant == "nohisto":
            fusion_dim = 3 * dim_h
        elif self.readout_variant == "node_only":
            fusion_dim = 2 * dim_h
        else:
            raise ValueError(
                "Unsupported OTFormer readout variant: " f"{self.readout_variant}"
            )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, readout_dim),
            nn.LayerNorm(readout_dim),
        )

        task_type = cfg.dataset.task_type
        self.is_classification = "classification" in task_type
        if self.is_classification:
            hidden = cfg.otformer.finetune.cls_hidden
            mlp_act_name = getattr(cfg.otformer.finetune, "cls_mlp_activation", "gelu")
            mlp_act = register.act_dict[mlp_act_name]()
            self.task_head = nn.Sequential(
                nn.Linear(readout_dim, hidden),
                mlp_act,
                nn.Dropout(cfg.otformer.dropout),
                nn.Linear(hidden, dim_out),
            )
            act_name = getattr(cfg.otformer.finetune, "cls_activation", "sigmoid")
            self.output_activation = self._get_output_activation(act_name)
        else:
            hidden = cfg.otformer.finetune.reg_hidden
            mlp_act_name = getattr(cfg.otformer.finetune, "reg_mlp_activation", "gelu")
            mlp_act = register.act_dict[mlp_act_name]()
            self.task_head = nn.Sequential(
                nn.Linear(readout_dim, hidden),
                mlp_act,
                nn.Dropout(cfg.otformer.dropout),
                nn.Linear(hidden, dim_out),
            )
            self.output_activation = nn.Identity()

    def _get_output_activation(self, name):
        name = name.lower()
        if name == "sigmoid":
            return nn.Sigmoid()
        elif name == "softmax":
            return nn.Softmax(dim=-1)
        elif name == "none" or name == "identity":
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported output activation: {name}")

    def _compute_ot_readout(self, transport, batch_vec):
        w_per_node = transport.sum(dim=1)
        return scatter_add(w_per_node, batch_vec, dim=0)

    def _compute_node_readout(self, h_out, batch_vec):
        mean_pooled = scatter_mean(h_out, batch_vec, dim=0)
        max_pooled, _ = scatter_max(h_out, batch_vec, dim=0)
        max_pooled = max_pooled.nan_to_num(0.0)
        return torch.cat([mean_pooled, max_pooled], dim=-1)

    def _compute_pair_readout(self, z_out, node_mask):
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        pooling = getattr(cfg.otformer.finetune, "pooling_z", "mean").lower()
        if pooling == "max":
            neg_inf = torch.finfo(z_out.dtype).min
            z_masked = z_out.masked_fill(~pair_mask.unsqueeze(-1), neg_inf)
            r_z = z_masked.amax(dim=(1, 2))
            r_z = torch.where(torch.isfinite(r_z), r_z, torch.zeros_like(r_z))
            return r_z

        z_masked = z_out * pair_mask.unsqueeze(-1).float()
        valid_counts = pair_mask.sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)
        return z_masked.sum(dim=(1, 2)) / valid_counts

    def forward(self, batch):
        h_out = batch.x
        aux = batch.otformer_aux
        z_out = aux["z_out"]
        node_mask = aux["node_mask"]
        batch_vec = batch.batch

        r_h = self._compute_node_readout(h_out, batch_vec)
        if self.readout_variant == "node_only":
            graph_embed = self.fusion(r_h)
        else:
            r_z = self._compute_pair_readout(z_out, node_mask)
            if self.readout_variant == "nohisto":
                graph_embed = self.fusion(torch.cat([r_h, r_z], dim=-1))
            else:
                transport = aux["transport"]
                r_ot = self._compute_ot_readout(transport, batch_vec)
                graph_embed = self.fusion(torch.cat([r_h, r_z, r_ot], dim=-1))
                aux["motif_hist_graph"] = r_ot

        logits = self.task_head(graph_embed)
        pred = logits
        if self.is_classification:
            batch.graph_pred_prob = self.output_activation(logits)

        batch.graph_feature = graph_embed
        return pred, batch.y
