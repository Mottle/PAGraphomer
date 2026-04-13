import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import new_layer_config, BatchNorm1dNode
from torch_geometric.graphgym.register import register_network

from gps.layer.gps_layer import GPSLayer


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """

    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
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
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if "PNA" in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_edge, -1, -1, has_act=False, has_bias=False, cfg=cfg
                    )
                )

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network("GPSModel")
class GPSModel(torch.nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        self.mask_token = None
        self.mask_decoder = None
        self.mask_loss = None
        self.edge_mask_token = None
        self.edge_decoder = None
        self.edge_loss = None

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split("+")
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(
                GPSLayer(
                    dim_h=cfg.gt.dim_hidden,
                    local_gnn_type=local_gnn_type,
                    global_model_type=global_model_type,
                    num_heads=cfg.gt.n_heads,
                    act=cfg.gnn.act,
                    pna_degrees=cfg.gt.pna_degrees,
                    equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                    dropout=cfg.gt.dropout,
                    attn_dropout=cfg.gt.attn_dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    bigbird_cfg=cfg.gt.bigbird,
                    log_attn_weights=cfg.train.mode == "log-attn-weights",
                )
            )
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        if getattr(cfg.gps.pretrain, "enable", False):
            num_atom_types = int(getattr(cfg.dataset, "node_encoder_num_types", 0))
            num_edge_types = int(getattr(cfg.dataset, "edge_encoder_num_types", 0))
            if num_atom_types < 1:
                raise ValueError(
                    "GPS masked pretraining requires dataset.node_encoder_num_types > 0"
                )
            if cfg.dataset.edge_encoder and num_edge_types < 1:
                raise ValueError(
                    "GPS edge masking requires dataset.edge_encoder_num_types > 0"
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

    def _sample_pretrain_mask(self, num_nodes, device):
        ratio = float(getattr(cfg.gps.pretrain, "mask_ratio", 0.15))
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

    def _forward_backbone(self, batch):
        batch = self.encoder(batch)
        if hasattr(self, "pre_mp"):
            batch = self.pre_mp(batch)
        batch = self.layers(batch)
        return batch

    def forward_pretrain(self, batch):
        if not cfg.dataset.node_encoder:
            raise RuntimeError(
                "GPS masked pretraining requires dataset.node_encoder=True."
            )
        if batch.x.dim() < 2:
            raise RuntimeError(
                "GPS masked pretraining expects raw categorical node features in batch.x."
            )

        atom_target = batch.x[:, 0].long()
        atom_mask = self._sample_pretrain_mask(atom_target.size(0), atom_target.device)
        edge_target = None
        edge_mask = None
        edge_supervision_idx = None
        if cfg.dataset.edge_encoder:
            if not hasattr(batch, "edge_attr") or batch.edge_attr is None:
                raise RuntimeError(
                    "GPS edge masking requires raw categorical edge_attr in the batch."
                )
            edge_target = batch.edge_attr.long()
            if edge_target.dim() > 1:
                edge_target = edge_target[:, 0]
            edge_mask, edge_supervision_idx = self._sample_edge_pretrain_mask(
                batch.edge_index,
                float(getattr(cfg.gps.pretrain, "edge_mask_ratio", 0.15)),
                atom_target.device,
            )

        batch = self.encoder(batch)
        batch.x = self._apply_mask_token(batch.x, atom_mask)
        if cfg.dataset.edge_encoder:
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
            float(getattr(cfg.gps.pretrain, "w_mask_atom", 1.0)) * atom_loss
            + float(getattr(cfg.gps.pretrain, "w_mask_edge", 1.0)) * edge_loss
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
        if getattr(cfg.gps.pretrain, "enable", False):
            return self.forward_pretrain(batch)

        batch = self._forward_backbone(batch)
        return self.post_mp(batch)
