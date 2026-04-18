import torch
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import BatchNorm1dNode, new_layer_config
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_mean


class FeatureEncoder(torch.nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    )
                )
            self.dim_in = cfg.gnn.dim_inner

        if cfg.posenc_RRWP.enable:
            rrwp_steps = int(
                getattr(
                    cfg.posenc_RRWP,
                    "ksteps",
                    getattr(cfg.posenc_RRWP, "walk_length", 8),
                )
            )
            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"](
                rrwp_steps,
                cfg.gnn.dim_inner,
            )

        if cfg.dataset.edge_encoder:
            cfg.gnn.dim_edge = cfg.gnn.dim_inner
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_edge,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    )
                )

        if cfg.posenc_RRWP.enable:
            rel_pe_dim = int(
                getattr(
                    cfg.posenc_RRWP,
                    "ksteps",
                    getattr(cfg.posenc_RRWP, "walk_length", 8),
                )
            )
            self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"](
                rel_pe_dim,
                cfg.gnn.dim_edge,
                pad_to_full_graph=cfg.gt.attn.full_attn,
                add_node_attr_as_self_loop=False,
                fill_value=0.0,
            )

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network("GritTransformer")
class GritTransformer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.ablation = True
        self.ablation = False
        self.mask_token = None
        self.mask_decoder = None
        self.mask_loss = None
        self.edge_mask_token = None
        self.edge_decoder = None
        self.edge_loss = None
        self.motil_diffusion_head = None
        self.motil_diffusion_attr_decoders = None
        self.motil_atom_attr_dims = ()
        self.motil_fg_node_proj = None
        self.motil_fg_msg_mlp = None
        self.motil_fg_update_norm = None
        self.motil_fg_readout = None
        self.motil_fg_local_depth = 0

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if cfg.gt.dim_hidden != cfg.gnn.dim_inner or cfg.gnn.dim_inner != dim_in:
            raise ValueError(
                "The inner and hidden dims must match for GRIT: "
                f"gt.dim_hidden={cfg.gt.dim_hidden}, "
                f"gnn.dim_inner={cfg.gnn.dim_inner}, dim_in={dim_in}"
            )

        TransformerLayer = register.layer_dict.get(cfg.gt.layer_type, None)
        if TransformerLayer is None:
            raise ValueError(f"Unknown GRIT layer type: {cfg.gt.layer_type}")

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(
                TransformerLayer(
                    in_dim=cfg.gt.dim_hidden,
                    out_dim=cfg.gt.dim_hidden,
                    num_heads=cfg.gt.n_heads,
                    dropout=cfg.gt.dropout,
                    act=cfg.gnn.act,
                    attn_dropout=cfg.gt.attn_dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    residual=True,
                    norm_e=cfg.gt.attn.norm_e,
                    O_e=cfg.gt.attn.O_e,
                    cfg=cfg.gt,
                )
            )
        self.layers = torch.nn.Sequential(*layers)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        if getattr(cfg.grit.pretrain, "enable", False):
            num_atom_types = int(getattr(cfg.dataset, "node_encoder_num_types", 0))
            num_edge_types = int(getattr(cfg.dataset, "edge_encoder_num_types", 0))
            if num_atom_types < 1:
                raise ValueError(
                    "GRIT masked pretraining requires dataset.node_encoder_num_types > 0"
                )
            if cfg.dataset.edge_encoder and num_edge_types < 1:
                raise ValueError(
                    "GRIT edge masking requires dataset.edge_encoder_num_types > 0"
                )
            self.mask_token = torch.nn.Parameter(torch.zeros(1, cfg.gnn.dim_inner))
            self.mask_decoder = torch.nn.Linear(cfg.gnn.dim_inner, num_atom_types)
            self.mask_loss = torch.nn.CrossEntropyLoss()
            if cfg.dataset.edge_encoder:
                self.edge_mask_token = torch.nn.Parameter(
                    torch.zeros(1, cfg.gnn.dim_edge)
                )
                self.edge_decoder = torch.nn.Linear(
                    2 * cfg.gnn.dim_inner, num_edge_types
                )
                self.edge_loss = torch.nn.CrossEntropyLoss()

        if getattr(cfg.grit.motil_pretrain, "enable", False):
            diffusion_steps = int(
                getattr(cfg.grit.motil_pretrain, "diffusion_steps", 1000)
            )
            if diffusion_steps < 1:
                raise ValueError(
                    "GRIT MotiL-style pretraining requires diffusion_steps >= 1"
                )
            diffusion_hidden = int(
                getattr(cfg.grit.motil_pretrain, "diffusion_hidden", cfg.gnn.dim_inner)
            )
            diffusion_act = register.act_dict[cfg.gnn.act]
            self.motil_diffusion_head = torch.nn.Sequential(
                torch.nn.Linear(cfg.gnn.dim_inner, diffusion_hidden),
                diffusion_act(),
                torch.nn.Linear(diffusion_hidden, cfg.gnn.dim_inner),
            )
            attr_dims = list(
                getattr(
                    cfg.grit.motil_pretrain,
                    "atom_attr_dims",
                    [119, 5, 12, 12, 10, 6, 6, 2, 2],
                )
            )
            if not attr_dims or any(int(dim) <= 1 for dim in attr_dims):
                raise ValueError(
                    "GRIT MotiL-style pretraining requires valid atom_attr_dims."
                )
            self.motil_atom_attr_dims = tuple(int(dim) for dim in attr_dims)
            self.motil_diffusion_attr_decoders = torch.nn.ModuleList(
                [
                    torch.nn.Linear(cfg.gnn.dim_inner, dim)
                    for dim in self.motil_atom_attr_dims
                ]
            )
            self.motil_fg_local_depth = int(
                getattr(cfg.grit.motil_pretrain, "fg_local_depth", 2)
            )
            if self.motil_fg_local_depth < 1:
                raise ValueError(
                    "GRIT MotiL-style pretraining requires fg_local_depth >= 1"
                )
            self.motil_fg_node_proj = torch.nn.Linear(
                cfg.gnn.dim_inner, cfg.gnn.dim_inner
            )
            self.motil_fg_msg_mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                diffusion_act(),
                torch.nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
            )
            self.motil_fg_update_norm = torch.nn.LayerNorm(cfg.gnn.dim_inner)
            self.motil_fg_readout = torch.nn.Sequential(
                torch.nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                diffusion_act(),
                torch.nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
            )
            betas = torch.linspace(1e-4, 0.02, diffusion_steps, dtype=torch.float32)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)
            self.register_buffer("motil_betas", betas, persistent=False)
            self.register_buffer("motil_alphas", alphas, persistent=False)
            self.register_buffer("motil_alpha_bars", alpha_bars, persistent=False)

    def _sample_pretrain_mask(self, num_nodes, device):
        ratio = float(getattr(cfg.grit.pretrain, "mask_ratio", 0.15))
        if ratio <= 0.0 or num_nodes == 0:
            return torch.zeros(num_nodes, dtype=torch.bool, device=device)

        mask = torch.rand(num_nodes, device=device) < ratio
        if not mask.any():
            mask[torch.randint(num_nodes, (1,), device=device)] = True
        return mask

    def _apply_mask_token(self, x, mask):
        if not mask.any():
            return x
        x = x.clone()
        x[mask] = self.mask_token.expand(int(mask.sum().item()), -1)
        return x

    def _sample_edge_pretrain_mask(self, edge_index, ratio, device):
        num_edges = edge_index.size(1)
        if ratio <= 0.0 or num_edges == 0:
            return torch.zeros(num_edges, dtype=torch.bool, device=device), torch.empty(
                0, dtype=torch.long, device=device
            )

        edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        undirected_groups = {}
        for eid in range(num_edges):
            src = int(edge_index[0, eid].item())
            dst = int(edge_index[1, eid].item())
            key = (src, dst) if src <= dst else (dst, src)
            undirected_groups.setdefault(key, []).append(eid)

        group_keys = list(undirected_groups.keys())
        group_mask = torch.rand(len(group_keys), device=device) < ratio
        if not group_mask.any() and len(group_keys) > 0:
            group_mask[torch.randint(len(group_keys), (1,), device=device)] = True

        supervised_edge_ids = []
        for group_id, key in enumerate(group_keys):
            if group_mask[group_id]:
                group_edge_ids = undirected_groups[key]
                edge_mask[group_edge_ids] = True
                supervised_edge_ids.append(group_edge_ids[0])

        if supervised_edge_ids:
            supervised_edge_ids = torch.tensor(
                supervised_edge_ids, dtype=torch.long, device=device
            )
        else:
            supervised_edge_ids = torch.empty(0, dtype=torch.long, device=device)
        return edge_mask, supervised_edge_ids

    def _apply_edge_mask_token(self, edge_attr, edge_mask):
        if edge_attr is None or self.edge_mask_token is None or not edge_mask.any():
            return edge_attr
        edge_attr = edge_attr.clone()
        edge_attr[edge_mask] = self.edge_mask_token.expand(
            int(edge_mask.sum().item()), -1
        )
        return edge_attr

    def _compute_pretrain_loss(self, atom_logits, atom_target, atom_mask):
        if atom_mask.any():
            return self.mask_loss(atom_logits[atom_mask], atom_target[atom_mask])
        return atom_logits.sum() * 0.0

    def _compute_edge_pretrain_loss(
        self, edge_logits, edge_target, edge_supervision_idx
    ):
        if (
            edge_logits is None
            or edge_supervision_idx is None
            or edge_supervision_idx.numel() == 0
        ):
            if edge_logits is None:
                return self.mask_token.sum() * 0.0
            return edge_logits.sum() * 0.0
        return self.edge_loss(
            edge_logits[edge_supervision_idx], edge_target[edge_supervision_idx]
        )

    def _decode_edge_types(self, node_repr, edge_index):
        if self.edge_decoder is None:
            return None
        src = node_repr[edge_index[0]]
        dst = node_repr[edge_index[1]]
        edge_repr = torch.cat((src + dst, torch.abs(src - dst)), dim=-1)
        return self.edge_decoder(edge_repr)

    def _encode_batch(self, batch):
        return self.encoder(batch)

    def _forward_encoder_pre_mp(self, batch):
        batch = self._encode_batch(batch)
        if hasattr(self, "pre_mp"):
            batch = self.pre_mp(batch)
        return batch

    def _clone_motil_view_batch(self, batch):
        cloned = batch.clone()
        if hasattr(cloned, "x") and torch.is_tensor(cloned.x):
            cloned.x = cloned.x.clone()
        if hasattr(cloned, "edge_attr") and torch.is_tensor(cloned.edge_attr):
            cloned.edge_attr = cloned.edge_attr.clone()
        return cloned

    def _forward_layers_with_dropout(
        self, batch, dropout_override=None, depth_override=None
    ):
        num_layers = len(self.layers)
        if depth_override is None:
            active_depth = num_layers
        else:
            active_depth = max(1, min(int(depth_override), num_layers))

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx >= active_depth:
                break
            if dropout_override is None:
                batch = layer(batch)
                continue
            prev_dropout = getattr(layer, "dropout", None)
            prev_attn_dropout = getattr(
                getattr(layer, "attention", None), "dropout", None
            )
            if prev_dropout is not None:
                layer.dropout = float(dropout_override)
            if prev_attn_dropout is not None:
                prev_attn_dropout.p = float(dropout_override)
            batch = layer(batch)
            if prev_dropout is not None:
                layer.dropout = prev_dropout
            if prev_attn_dropout is not None:
                prev_attn_dropout.p = getattr(
                    cfg.gt, "attn_dropout", prev_attn_dropout.p
                )
        return batch

    def _forward_backbone(self, batch):
        batch = self._forward_encoder_pre_mp(batch)
        return self._forward_layers_with_dropout(batch)

    def _forward_backbone_with_dropout(
        self, batch, dropout_override=None, depth_override=None
    ):
        batch = self._forward_encoder_pre_mp(batch)
        return self._forward_layers_with_dropout(
            batch,
            dropout_override=dropout_override,
            depth_override=depth_override,
        )

    def _pointwise_contrastive_loss(self, z_i, z_j):
        if z_i.size(0) == 0 or z_j.size(0) == 0:
            return z_i.sum() * 0.0
        batch_size = z_i.size(0)
        feature_dim = z_i.size(1)
        eps = 1e-15
        lambda_ = 1.0 / feature_dim
        z1_norm = (z_i - z_i.mean(dim=0)) / (z_i.std(dim=0) + eps)
        z2_norm = (z_j - z_j.mean(dim=0)) / (z_j.std(dim=0) + eps)
        corr = (z1_norm.T @ z2_norm) / batch_size
        off_diagonal_mask = ~torch.eye(
            feature_dim, device=corr.device, dtype=torch.bool
        )
        loss = (1 - corr.diagonal()).pow(2).sum()
        loss = loss + lambda_ * corr[off_diagonal_mask].pow(2).sum()
        return loss

    def _build_fg_motif_context(self, batch, node_repr):
        required = ["fg_count", "fg_atom_index", "fg_atom_fg_id", "fg_type"]
        if any(not hasattr(batch, attr) for attr in required):
            return None

        device = node_repr.device
        fg_count = batch.fg_count.view(-1).to(device=device, dtype=torch.long)
        if fg_count.numel() == 0:
            return None

        total_fg = int(fg_count.sum().item())
        if total_fg == 0:
            return None

        fg_type = batch.fg_type.to(device=device, dtype=torch.long)
        if fg_type.numel() != total_fg:
            return None

        atom_node_idx = batch.fg_atom_index.to(device=device, dtype=torch.long)
        atom_fg_local = batch.fg_atom_fg_id.to(device=device, dtype=torch.long)
        if atom_node_idx.numel() == 0 or atom_fg_local.numel() == 0:
            return None

        fg_offsets = torch.cumsum(
            torch.cat([fg_count.new_zeros(1), fg_count[:-1]], dim=0), dim=0
        )
        atom_graph_id = batch.batch[atom_node_idx].to(device=device, dtype=torch.long)
        atom_fg_global = atom_fg_local + fg_offsets[atom_graph_id]
        if atom_fg_global.numel() == 0:
            return None
        if int(atom_fg_global.max().item()) >= total_fg:
            return None

        total_fg = int(total_fg)
        fg_offsets = fg_offsets.to(device=device, dtype=torch.long)
        fg_start_pos = torch.cumsum(
            torch.bincount(atom_fg_global, minlength=total_fg), dim=0
        )
        fg_start_pos = fg_start_pos - torch.bincount(atom_fg_global, minlength=total_fg)

        edge_src_local = atom_fg_global.new_empty((0,))
        edge_dst_local = atom_fg_global.new_empty((0,))
        if (
            hasattr(batch, "fg_edge_index")
            and hasattr(batch, "fg_edge_fg_id")
            and hasattr(batch, "fg_edge_src_pos")
            and hasattr(batch, "fg_edge_dst_pos")
            and batch.fg_edge_index.numel() > 0
            and batch.fg_edge_fg_id.numel() > 0
        ):
            fg_edge_index = batch.fg_edge_index.to(device=device, dtype=torch.long)
            edge_src_node = fg_edge_index[0]
            edge_dst_node = fg_edge_index[1]
            edge_fg_local = batch.fg_edge_fg_id.to(device=device, dtype=torch.long)
            edge_graph_id = batch.batch[edge_src_node].to(
                device=device, dtype=torch.long
            )
            edge_fg_global = edge_fg_local + fg_offsets[edge_graph_id]
            edge_src_pos = batch.fg_edge_src_pos.to(device=device, dtype=torch.long)
            edge_dst_pos = batch.fg_edge_dst_pos.to(device=device, dtype=torch.long)
            valid_edge = (edge_fg_global >= 0) & (edge_fg_global < total_fg)
            if valid_edge.any():
                edge_src_local = (
                    fg_start_pos[edge_fg_global[valid_edge]] + edge_src_pos[valid_edge]
                )
                edge_dst_local = (
                    fg_start_pos[edge_fg_global[valid_edge]] + edge_dst_pos[valid_edge]
                )

        return {
            "total_fg": total_fg,
            "fg_type": fg_type,
            "atom_node_idx": atom_node_idx,
            "atom_fg_global": atom_fg_global,
            "edge_src_local": edge_src_local,
            "edge_dst_local": edge_dst_local,
        }

    def _encode_fg_motif_instances(self, node_repr, batch, context=None):
        if context is None:
            context = self._build_fg_motif_context(batch, node_repr)
        if context is None:
            return node_repr.new_zeros((0, node_repr.size(-1))), node_repr.new_zeros(
                (0,), dtype=torch.long
            )

        motif_node_repr = self.motil_fg_node_proj(node_repr[context["atom_node_idx"]])

        edge_src_local = context["edge_src_local"]
        edge_dst_local = context["edge_dst_local"]
        if edge_src_local.numel() > 0 and edge_dst_local.numel() > 0:
            for _ in range(self.motil_fg_local_depth):
                msg_in = torch.cat(
                    [motif_node_repr[edge_src_local], motif_node_repr[edge_dst_local]],
                    dim=-1,
                )
                messages = self.motil_fg_msg_mlp(msg_in)
                aggregated = scatter_mean(
                    messages,
                    edge_dst_local,
                    dim=0,
                    dim_size=motif_node_repr.size(0),
                )
                motif_node_repr = self.motil_fg_update_norm(
                    motif_node_repr + aggregated
                )

        motif_embeddings = scatter_mean(
            motif_node_repr,
            context["atom_fg_global"],
            dim=0,
            dim_size=context["total_fg"],
        )
        motif_embeddings = self.motil_fg_readout(motif_embeddings)
        return motif_embeddings, context["fg_type"]

    def _compute_motil_pretrain_loss(self, batch):
        if self.motil_diffusion_head is None:
            raise RuntimeError("GRIT MotiL-style pretraining head is not initialized")

        phase = str(getattr(batch, "grit_pretrain_phase", "joint"))
        x_raw = batch.x.clone()
        zero = torch.zeros((), device=x_raw.device, dtype=torch.float32)
        if phase in {"diffusion", "joint"}:
            encoded = self._encode_batch(batch.clone())
            h0 = encoded.x
            if x_raw.dim() < 2:
                raise RuntimeError(
                    "GRIT MotiL diffusion expects raw node attributes in batch.x."
                )
            num_attrs = len(self.motil_atom_attr_dims)
            if x_raw.size(1) < num_attrs:
                raise RuntimeError(
                    "GRIT MotiL diffusion received fewer atom attributes than expected: "
                    f"got={x_raw.size(1)}, expected_at_least={num_attrs}."
                )
            atom_targets = x_raw[:, :num_attrs].long()
            diffusion_steps = int(self.motil_alpha_bars.numel())
            t = torch.randint(0, diffusion_steps, (h0.size(0),), device=h0.device)
            alpha_bar_t = self.motil_alpha_bars[t].unsqueeze(-1)
            noise = torch.randn_like(h0)
            h_t = torch.sqrt(alpha_bar_t) * h0 + torch.sqrt(1 - alpha_bar_t) * noise
            pred_noise = self.motil_diffusion_head(h_t)
            h0_pred = (h_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(
                alpha_bar_t
            )
            attr_losses = []
            for idx, decoder in enumerate(self.motil_diffusion_attr_decoders):
                target = atom_targets[:, idx]
                logits = decoder(h0_pred)
                if target.numel() > 0:
                    min_target = int(target.min().item())
                    max_target = int(target.max().item())
                    if min_target < 0 or max_target >= logits.size(-1):
                        raise RuntimeError(
                            "GRIT MotiL diffusion atom attribute targets are out of "
                            f"range at attr_idx={idx}: min={min_target}, "
                            f"max={max_target}, num_classes={logits.size(-1)}."
                        )
                attr_losses.append(F.cross_entropy(logits, target))
            diffusion_loss = torch.stack(attr_losses).mean()
        else:
            diffusion_loss = zero

        active_task = getattr(batch, "grit_pretrain_task", "contrast_mol")
        active_task = str(active_task)

        graph_emb1 = None
        graph_emb2 = None
        fg_emb1 = None
        fg_emb2 = None
        contrast_loss = diffusion_loss.new_zeros(())
        fg_contrast_loss = diffusion_loss.new_zeros(())

        if phase in {"contrast", "joint"}:
            if phase == "contrast" and active_task not in {
                "contrast_mol",
                "contrast_fgs",
            }:
                raise ValueError(f"Unexpected GRIT MotiL pretrain task: {active_task}")

            dropout1 = float(getattr(cfg.grit.motil_pretrain, "dropout1", 0.3))
            dropout2 = float(getattr(cfg.grit.motil_pretrain, "dropout2", 0.3))
            depth1 = getattr(cfg.grit.motil_pretrain, "depth1", -1)
            depth2 = getattr(cfg.grit.motil_pretrain, "depth2", -1)
            depth1 = None if int(depth1) < 0 else int(depth1)
            depth2 = None if int(depth2) < 0 else int(depth2)

            encoded_shared = self._forward_encoder_pre_mp(batch.clone())
            out1 = self._forward_layers_with_dropout(
                self._clone_motil_view_batch(encoded_shared),
                dropout_override=dropout1,
                depth_override=depth1,
            )
            out2 = self._forward_layers_with_dropout(
                self._clone_motil_view_batch(encoded_shared),
                dropout_override=dropout2,
                depth_override=depth2,
            )

            if phase == "joint" or active_task == "contrast_mol":
                graph_emb1 = global_mean_pool(out1.x, out1.batch)
                graph_emb2 = global_mean_pool(out2.x, out2.batch)
                contrast_loss = self._pointwise_contrastive_loss(graph_emb1, graph_emb2)

            if phase == "joint" or active_task == "contrast_fgs":
                shared_fg_context = self._build_fg_motif_context(out1, out1.x)
                fg_instances1, fg_type1 = self._encode_fg_motif_instances(
                    out1.x, out1, context=shared_fg_context
                )
                fg_instances2, fg_type2 = self._encode_fg_motif_instances(
                    out2.x, out2, context=shared_fg_context
                )

                if fg_instances1.size(0) > 0 and fg_instances2.size(0) > 0:
                    num_fg_types = int(
                        max(fg_type1.max().item(), fg_type2.max().item()) + 1
                    )
                    fg_prototypes1 = scatter_mean(
                        fg_instances1,
                        fg_type1,
                        dim=0,
                        dim_size=num_fg_types,
                    )
                    fg_prototypes2 = scatter_mean(
                        fg_instances2,
                        fg_type2,
                        dim=0,
                        dim_size=num_fg_types,
                    )
                    fg_counts1 = torch.bincount(fg_type1, minlength=num_fg_types)
                    fg_counts2 = torch.bincount(fg_type2, minlength=num_fg_types)
                    common_type_mask = (fg_counts1 > 0) & (fg_counts2 > 0)
                    fg_emb1 = fg_prototypes1[common_type_mask]
                    fg_emb2 = fg_prototypes2[common_type_mask]
                else:
                    fg_emb1 = diffusion_loss.new_zeros((0, out1.x.size(-1)))
                    fg_emb2 = diffusion_loss.new_zeros((0, out2.x.size(-1)))

                fg_contrast_loss = self._pointwise_contrastive_loss(fg_emb1, fg_emb2)

        total_loss = zero
        if phase in {"diffusion", "joint"}:
            total_loss = (
                total_loss
                + float(getattr(cfg.grit.motil_pretrain, "w_diffusion", 1.0))
                * diffusion_loss
            )
        if phase == "joint":
            total_loss = (
                total_loss
                + float(getattr(cfg.grit.motil_pretrain, "w_contrast", 1.0))
                * contrast_loss
            )
            total_loss = (
                total_loss
                + float(getattr(cfg.grit.motil_pretrain, "w_fgs", 1.0))
                * fg_contrast_loss
            )
        elif phase == "contrast":
            if active_task == "contrast_mol":
                total_loss = (
                    total_loss
                    + float(getattr(cfg.grit.motil_pretrain, "w_contrast", 1.0))
                    * contrast_loss
                )
            else:
                total_loss = (
                    total_loss
                    + float(getattr(cfg.grit.motil_pretrain, "w_fgs", 1.0))
                    * fg_contrast_loss
                )

        batch.grit_aux = {
            "loss": total_loss,
            "phase": phase,
            "active_task": active_task,
            "losses": {
                "diffusion": diffusion_loss,
                "contrast_mol": contrast_loss,
                "contrast_fgs": fg_contrast_loss,
            },
            "graph_emb1": graph_emb1,
            "graph_emb2": graph_emb2,
            "fg_emb1": fg_emb1,
            "fg_emb2": fg_emb2,
        }
        true = (
            batch.y
            if hasattr(batch, "y") and batch.y is not None
            else x_raw.new_zeros((1, 1), dtype=torch.float32)
        )
        pred = true.float().new_zeros(true.shape)
        return pred, true

    def forward_pretrain(self, batch):
        if not cfg.dataset.node_encoder:
            raise RuntimeError(
                "GRIT masked pretraining requires dataset.node_encoder=True."
            )
        if batch.x.dim() < 2:
            raise RuntimeError(
                "GRIT masked pretraining expects raw categorical node features in batch.x."
            )

        atom_target = batch.x[:, 0].long()
        atom_mask = self._sample_pretrain_mask(atom_target.size(0), atom_target.device)
        edge_target = None
        edge_mask = None
        edge_supervision_idx = None
        if cfg.dataset.edge_encoder:
            if not hasattr(batch, "edge_attr") or batch.edge_attr is None:
                raise RuntimeError(
                    "GRIT edge masking requires raw categorical edge_attr in the batch."
                )
            # RRWP relative edge encoding expands the sparse graph to a padded/full
            # graph representation, so raw edge masks no longer align with the
            # encoded edge set. Keep GRIT pretraining stable by disabling edge
            # masking whenever RRWP is active.
            if not cfg.posenc_RRWP.enable:
                edge_target = batch.edge_attr.long()
                if edge_target.dim() > 1:
                    edge_target = edge_target[:, 0]
                edge_mask, edge_supervision_idx = self._sample_edge_pretrain_mask(
                    batch.edge_index,
                    float(getattr(cfg.grit.pretrain, "edge_mask_ratio", 0.15)),
                    atom_target.device,
                )

        batch = self.encoder(batch)
        batch.x = self._apply_mask_token(batch.x, atom_mask)
        if cfg.dataset.edge_encoder and edge_mask is not None:
            batch.edge_attr = self._apply_edge_mask_token(batch.edge_attr, edge_mask)
        if hasattr(self, "pre_mp"):
            batch = self.pre_mp(batch)
        batch = self.layers(batch)

        atom_logits = self.mask_decoder(batch.x)
        edge_logits = self._decode_edge_types(batch.x, batch.edge_index)
        atom_loss = self._compute_pretrain_loss(atom_logits, atom_target, atom_mask)
        edge_loss = self._compute_edge_pretrain_loss(
            edge_logits, edge_target, edge_supervision_idx
        )
        loss = (
            float(getattr(cfg.grit.pretrain, "w_mask_atom", 1.0)) * atom_loss
            + float(getattr(cfg.grit.pretrain, "w_mask_edge", 1.0)) * edge_loss
        )
        batch.gps_aux = {
            "loss": loss,
            "losses": {
                "mask_atom": atom_loss,
                "mask_edge": edge_loss,
            },
            "atom_logits": atom_logits,
            "atom_target": atom_target,
            "atom_mask": atom_mask,
            "edge_logits": edge_logits,
            "edge_target": edge_target,
            "edge_mask": edge_mask,
            "edge_supervision_idx": edge_supervision_idx,
        }

        if hasattr(batch, "y") and batch.y is not None:
            true = batch.y
        else:
            true = atom_logits.new_zeros((1, 1))
        pred = true.float().new_zeros(true.shape)
        return pred, true

    def forward(self, batch):
        if getattr(cfg.grit.motil_pretrain, "enable", False):
            return self._compute_motil_pretrain_loss(batch)
        if getattr(cfg.grit.pretrain, "enable", False):
            return self.forward_pretrain(batch)
        for module in self.children():
            batch = module(batch)
        return batch
