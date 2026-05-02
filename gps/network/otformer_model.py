import random
import warnings
from collections import deque

import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import to_dense_adj

from gps.layer.otformer_layer import OTMotifMemory, OTFormerBlock, build_pair_init
from gps.layer.rum.models import RUMModel
from gps.network.gps_model import FeatureEncoder


def _infer_atom_targets(raw_x):
    if raw_x.dim() == 1:
        return raw_x.long()
    if raw_x.dim() == 2 and raw_x.size(1) == 1:
        return raw_x.squeeze(-1).long()
    if raw_x.dim() == 2:
        return raw_x.argmax(dim=-1).long()
    raise ValueError(
        f"Unsupported raw node feature shape for atom targets: {raw_x.shape}"
    )


def _infer_edge_type_targets(raw_edge_attr):
    if raw_edge_attr is None:
        return None
    if raw_edge_attr.dim() == 1:
        return raw_edge_attr.long()
    if raw_edge_attr.dim() == 2 and raw_edge_attr.size(1) == 1:
        return raw_edge_attr.squeeze(-1).long()
    if raw_edge_attr.dim() == 2:
        return raw_edge_attr.argmax(dim=-1).long()
    raise ValueError(
        "Unsupported raw edge feature shape for edge type targets: "
        f"{raw_edge_attr.shape}"
    )


def _local_node_index(node_batch):
    idx = torch.zeros_like(node_batch)
    num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 0
    for g in range(num_graphs):
        nodes = (node_batch == g).nonzero(as_tuple=False).flatten()
        idx[nodes] = torch.arange(nodes.numel(), device=node_batch.device)
    return idx


@register_network("OTFormerModel")
class OTFormerModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        if cfg.gnn.dim_inner != dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: dim_inner={cfg.gnn.dim_inner}, dim_in={dim_in}"
            )

        dim_h = cfg.gnn.dim_inner
        self.dim_h = dim_h
        dim_z = getattr(cfg.otformer, "dim_z", None)
        if dim_z is not None:
            dim_z = int(dim_z)
        else:
            dim_z = dim_h
        self.dim_z = dim_z
        self.recycling_iters = cfg.otformer.recycling_iters
        self.detach_recycle = cfg.otformer.detach_recycle
        self.ablation_enable = bool(getattr(cfg.otformer.ablation, "enable", False))
        self.disable_rum_ot = bool(
            getattr(cfg.otformer.ablation, "disable_rum_ot", False)
        )

        self.rum = RUMModel(
            in_features=dim_h,
            out_features=dim_h,
            hidden_features=dim_h,
            edge_features=dim_h,
            depth=cfg.otformer.rum.depth,
            num_samples=cfg.otformer.rum.num_samples,
            length=cfg.otformer.rum.rw_length,
            dropout=cfg.otformer.rum.dropout,
            binary=cfg.otformer.rum.binary,
            self_supervise=False,
            output_softmax=getattr(cfg.otformer.rum, "output_softmax", True),
        )
        self.ot_memory = OTMotifMemory(dim_h, cfg.otformer.motif.memory_size)
        self.motif_to_node = nn.Linear(dim_h, dim_h)
        self.path_to_node = nn.Linear(dim_h, dim_h)
        self.edge_proj = nn.Linear(dim_h, dim_h)
        self.pair_init_proj = nn.Linear(dim_h + 1, dim_z)
        self.use_spd = bool(getattr(cfg.otformer.pair, "use_spd", True))
        self.spd_max_dist = max(1, int(getattr(cfg.otformer.pair, "spd_max_dist", 16)))
        self.use_rrwp = bool(getattr(cfg.otformer.pair, "use_rrwp", False))
        self.rrwp_attr_name = getattr(cfg.otformer.pair, "rrwp_attr_name", "rrwp")
        rrwp_cfg_dim = int(getattr(cfg.otformer.pair, "rrwp_dim", 0))
        if self.use_rrwp and rrwp_cfg_dim <= 0 and hasattr(cfg, "posenc_RRWP"):
            rrwp_cfg_dim = int(getattr(cfg.posenc_RRWP, "walk_length", 0))
        self.rrwp_dim = rrwp_cfg_dim

        self.blocks = nn.ModuleList(
            [
                OTFormerBlock(
                    dim_h=dim_h,
                    dim_z=dim_z,
                    num_heads=cfg.otformer.num_heads,
                    dropout=cfg.otformer.dropout,
                    attn_dropout=cfg.otformer.attn_dropout,
                    layer_norm=cfg.otformer.layer_norm,
                    use_triangle=cfg.otformer.pair.use_triangle,
                    triangle_hidden=cfg.otformer.pair.triangle_hidden,
                    ffn_activation=getattr(cfg.otformer, "ffn_activation", "gelu"),
                    use_rrwp=self.use_rrwp,
                    rrwp_dim=self.rrwp_dim,
                )
                for _ in range(cfg.otformer.layers)
            ]
        )
        self.h_recycle_norm = nn.LayerNorm(dim_h)
        self.z_recycle_norm = nn.LayerNorm(dim_z)

        atom_classes = getattr(cfg.dataset, "node_encoder_num_types", 64)
        self.mask_token = nn.Parameter(torch.zeros(dim_h))
        nn.init.normal_(self.mask_token, std=0.02)
        self.atom_decoder = nn.Linear(dim_h, atom_classes)
        self.motif_decoder = nn.Linear(dim_h, cfg.otformer.motif.memory_size)
        denoise_mode = getattr(cfg.otformer.pretrain, "edge_denoise_mode", "random")
        if denoise_mode == "reconstruct":
            edge_type_classes = max(
                getattr(cfg.dataset, "edge_encoder_num_types", 4), 1
            )
            self.edge_mask_token = nn.Parameter(torch.zeros(dim_h))
            nn.init.normal_(self.edge_mask_token, std=0.02)
            self.edge_decoder = nn.Linear(dim_z, edge_type_classes)
        elif denoise_mode == "edge_type":
            n_edge_types = max(getattr(cfg.dataset, "edge_encoder_num_types", 4), 1)
            self.edge_type_noedge_id = n_edge_types
            self.edge_decoder = nn.Linear(dim_z, n_edge_types + 1)
        else:
            self.edge_decoder = nn.Linear(dim_z, 1)

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        if self.recycling_iters == 0 and len(self.blocks) > 0:
            warnings.warn(
                "recycling_iters=0 means Transformer blocks are never called. "
                "Set recycling_iters >= 1 to enable the transformer."
            )

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=dim_h, dim_out=dim_out)

        finetune_on = getattr(cfg.otformer.finetune, "enable", False)
        if finetune_on and cfg.gnn.head == "otformer_finetune":
            from gps.head.otformer_finetune_head import OTFormerFineTuneHead

            self.post_mp = OTFormerFineTuneHead(dim_in=dim_h, dim_out=dim_out)

    @staticmethod
    def _connected_components_from_edges(active_local_idx, edge_src, edge_dst):
        """Return connected components over `active_local_idx` using local edges."""
        if active_local_idx.numel() == 0:
            return []
        if active_local_idx.numel() == 1 or edge_src.numel() == 0:
            return [active_local_idx]

        max_local = int(active_local_idx.max().item()) + 1
        comp_map = torch.full(
            (max_local,),
            -1,
            dtype=torch.long,
            device=active_local_idx.device,
        )
        comp_map[active_local_idx] = torch.arange(
            active_local_idx.numel(), device=active_local_idx.device
        )
        src = comp_map[edge_src]
        dst = comp_map[edge_dst]
        valid = (src >= 0) & (dst >= 0) & (src != dst)
        src = src[valid]
        dst = dst[valid]

        n = int(active_local_idx.numel())
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in zip(src.tolist(), dst.tolist()):
            union(a, b)

        groups = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)
        return [
            active_local_idx[
                torch.tensor(members, device=active_local_idx.device, dtype=torch.long)
            ]
            for members in groups.values()
        ]

    def _sample_motif_block_mask(self, node_batch, edge_index, motif_id):
        device = node_batch.device
        n_nodes = node_batch.size(0)
        mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        num_graphs = int(node_batch.max().item()) + 1 if n_nodes > 0 else 0

        ratio = cfg.otformer.pretrain.motif_mask_ratio
        if ratio <= 0:
            return mask
        topk = cfg.otformer.pretrain.motif_topk
        for g in range(num_graphs):
            nodes = (node_batch == g).nonzero(as_tuple=False).flatten()
            if nodes.numel() == 0:
                continue
            local_idx = torch.full((n_nodes,), -1, device=device, dtype=torch.long)
            local_idx[nodes] = torch.arange(nodes.numel(), device=device)
            eids = (node_batch[edge_index[0]] == g).nonzero(as_tuple=False).flatten()
            if eids.numel() > 0:
                edge_src_local = local_idx[edge_index[0, eids]]
                edge_dst_local = local_idx[edge_index[1, eids]]
                valid_edges = (
                    (edge_src_local >= 0)
                    & (edge_dst_local >= 0)
                    & (edge_src_local != edge_dst_local)
                )
                edge_src_local = edge_src_local[valid_edges]
                edge_dst_local = edge_dst_local[valid_edges]
            else:
                edge_src_local = torch.empty(0, device=device, dtype=torch.long)
                edge_dst_local = torch.empty(0, device=device, dtype=torch.long)
            motif_g = motif_id[nodes]
            uniq, cnt = torch.unique(motif_g, return_counts=True)
            k = min(int(topk), uniq.numel())
            if k == 0:
                continue
            top = uniq[torch.topk(cnt, k=k).indices]
            for motif in top.tolist():
                motif_local_idx = (motif_g == motif).nonzero(as_tuple=False).flatten()
                if motif_local_idx.numel() == 0:
                    continue
                motif_edge_mask = (motif_g[edge_src_local] == motif) & (
                    motif_g[edge_dst_local] == motif
                )
                comps = self._connected_components_from_edges(
                    active_local_idx=motif_local_idx,
                    edge_src=edge_src_local[motif_edge_mask],
                    edge_dst=edge_dst_local[motif_edge_mask],
                )
                for comp_local in comps:
                    if torch.rand(1, device=device).item() < ratio:
                        mask[nodes[comp_local]] = True
        return mask

    def _perturb_edges(self, edge_index, edge_attr, node_batch, edge_type_target=None):
        if edge_attr is not None:
            edge_attr = edge_attr.clone()
        if node_batch.numel() == 0:
            return edge_index, edge_attr, None
        ratio = cfg.otformer.pretrain.edge_perturb_ratio
        device = edge_index.device
        num_graphs = node_batch.max().item() + 1
        edge_graph = node_batch[edge_index[0]]
        keep = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
        perturb_len_total = 0
        perturb_tensors = {}  # "b", "i", "j", "y" -> list of tensors

        mode = getattr(cfg.otformer.pretrain, "edge_denoise_mode", "random")
        reconstruct_mode = mode == "reconstruct"
        edge_type_mode = mode == "edge_type"
        collect_type_target = reconstruct_mode or edge_type_mode

        attr_work = edge_attr.clone() if edge_attr is not None else None
        n_total_nodes = (
            node_batch.numel()
        )  # total nodes across all graphs, NOT max graph index
        global_to_local = torch.full(
            (n_total_nodes,), -1, device=device, dtype=torch.long
        )
        node_counts = torch.zeros(num_graphs, device=device, dtype=torch.long)
        drop_per_graph = torch.zeros(num_graphs, device=device, dtype=torch.long)

        for g in range(num_graphs):
            nodes = (node_batch == g).nonzero(as_tuple=False).flatten()
            ng = nodes.numel()
            if ng < 2:
                continue
            node_counts[g] = ng
            global_to_local[nodes] = torch.arange(ng, device=device)

        for g in range(num_graphs):
            ng = int(node_counts[g].item())
            if ng < 2:
                continue
            nodes = node_batch == g
            mask = edge_graph == g
            eids = mask.nonzero(as_tuple=False).flatten()
            ne = eids.numel()
            if ne == 0:
                continue

            # --- Vectorized undirected pair grouping ---
            src_g = edge_index[0, eids]
            dst_g = edge_index[1, eids]
            mn = torch.min(src_g, dst_g)
            mx = torch.max(src_g, dst_g)
            pair_key = mn * n_total_nodes + mx
            sort_idx = torch.argsort(pair_key)
            sorted_key = pair_key[sort_idx]
            sorted_eids = eids[sort_idx]
            _, inv, cnt = torch.unique_consecutive(
                sorted_key, return_inverse=True, return_counts=True
            )
            _cumsum = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.long),
                    torch.cumsum(cnt, dim=0)[:-1],
                ]
            )
            n_undirected = inv.max().item() + 1 if inv.numel() > 0 else 0
            if n_undirected == 0:
                continue

            n_drop = max(1, int(n_undirected * ratio))
            n_drop = min(n_drop, n_undirected)
            drop_per_graph[g] = int(n_drop)
            perm = torch.randperm(n_undirected, device=device)[:n_drop]

            # Collect keep flags and perturb labels in vectorized form
            drop_flags = torch.zeros(n_undirected, dtype=torch.bool, device=device)
            drop_flags[perm] = True
            eid_drop_flags = drop_flags[inv]

            if not reconstruct_mode:
                keep[sorted_eids] = ~eid_drop_flags
            if reconstruct_mode and edge_attr is not None:
                attr_work[sorted_eids[eid_drop_flags]] = self.edge_mask_token.to(
                    attr_work.dtype
                )

            # Build perturbation labels for dropped pairs
            n_pos = n_drop * 2  # bidirectional for non-self-loop, 1 for self-loop
            b_arr = torch.full((n_pos,), g, device=device, dtype=torch.long)
            i_arr = torch.zeros(n_pos, device=device, dtype=torch.long)
            j_arr = torch.zeros(n_pos, device=device, dtype=torch.long)
            y_arr = torch.ones(n_pos, device=device, dtype=torch.long)
            pos = 0
            for p_idx in perm.tolist():
                start = int(_cumsum[p_idx].item())
                count = int(cnt[p_idx].item())
                eid0 = sorted_eids[start].item()
                s = int(edge_index[0, eid0].item())
                d = int(edge_index[1, eid0].item())
                li = int(global_to_local[s].item())
                lj = int(global_to_local[d].item())
                if s == d:
                    i_arr[pos] = li
                    j_arr[pos] = lj
                    pos += 1
                else:
                    i_arr[pos] = li
                    j_arr[pos] = lj
                    i_arr[pos + 1] = lj
                    j_arr[pos + 1] = li
                    pos += 2

            perturb_tensors.setdefault("b", []).append(b_arr[:pos])
            perturb_tensors.setdefault("i", []).append(i_arr[:pos])
            perturb_tensors.setdefault("j", []).append(j_arr[:pos])
            perturb_tensors.setdefault("y", []).append(y_arr[:pos])
            perturb_len_total += pos

            # Type targets
            if collect_type_target and edge_type_target is not None:
                t_arr = torch.zeros(pos, device=device, dtype=torch.long)
                tp = 0
                for p_idx in perm.tolist():
                    start = int(_cumsum[p_idx].item())
                    eid0 = sorted_eids[start].item()
                    s = int(edge_index[0, eid0].item())
                    et = int(edge_type_target[eid0].item())
                    if s == edge_index[1, eid0].item():
                        t_arr[tp] = et
                        tp += 1
                    else:
                        t_arr[tp] = et
                        t_arr[tp + 1] = et
                        tp += 2
                perturb_tensors.setdefault("target_type", []).append(t_arr[:tp])
        # --- End per-graph loop ---

        edge_keep = edge_index[:, keep]
        attr_keep = attr_work[keep] if attr_work is not None else None

        max_spd = getattr(cfg.otformer.pretrain, "hard_neg_max_spd", 3)

        add_triplets = []
        add_attr_list = []
        if edge_attr is not None:
            zero_attr = torch.zeros(
                edge_attr.size(1), device=device, dtype=edge_attr.dtype
            )

        if not reconstruct_mode:
            for g in range(num_graphs):
                nodes_t = (node_batch == g).nonzero(as_tuple=False).flatten()
                if nodes_t.numel() < 2:
                    continue
                nodes_list = nodes_t.tolist()
                eids_true = (edge_graph == g).nonzero(as_tuple=False).flatten()
                true_set = set()
                for eid in eids_true.tolist():
                    s = int(edge_index[0, eid].item())
                    d = int(edge_index[1, eid].item())
                    true_set.add((s, d) if s <= d else (d, s))
                keep_graph = edge_graph[keep]
                eids_keep = (keep_graph == g).nonzero(as_tuple=False).flatten()
                existing = set()
                for eid in eids_keep.tolist():
                    s = int(edge_keep[0, eid].item())
                    d = int(edge_keep[1, eid].item())
                    existing.add((s, d) if s <= d else (d, s))
                base_add = int(drop_per_graph[g].item())
                target_add = int(base_add * cfg.otformer.pretrain.edge_neg_ratio)
                if (
                    target_add <= 0
                    and base_add > 0
                    and cfg.otformer.pretrain.edge_neg_ratio > 0
                ):
                    target_add = 1
                if target_add == 0:
                    continue

                if mode == "hard_spd":
                    candidates = self._sample_hard_negatives_spd(
                        edge_keep, node_batch, g, nodes_t, existing | true_set, max_spd
                    )
                else:
                    candidates = self._sample_random_negatives(
                        nodes_t, existing | true_set, target_add
                    )

                added = 0
                for s, d in candidates:
                    if added >= target_add:
                        break
                    existing.add((s, d) if s <= d else (d, s))
                    add_triplets.extend([(s, d), (d, s)])
                    if edge_attr is not None:
                        add_attr_list.append(zero_attr.clone())
                        add_attr_list.append(zero_attr.clone())
                    li = int((nodes_t == s).nonzero().item())
                    lj = int((nodes_t == d).nonzero().item())
                    perturb_tensors.setdefault("b", []).append(
                        torch.tensor([g, g], device=device, dtype=torch.long)
                    )
                    perturb_tensors.setdefault("i", []).append(
                        torch.tensor([li, lj], device=device, dtype=torch.long)
                    )
                    perturb_tensors.setdefault("j", []).append(
                        torch.tensor([lj, li], device=device, dtype=torch.long)
                    )
                    perturb_tensors.setdefault("y", []).append(
                        torch.zeros(2, device=device, dtype=torch.long)
                    )
                    perturb_len_total += 2
                    if edge_type_mode:
                        noedge_id = int(getattr(self, "edge_type_noedge_id", 0))
                        perturb_tensors.setdefault("target_type", []).append(
                            torch.full((2,), noedge_id, device=device, dtype=torch.long)
                        )
                    added += 1

        if add_triplets:
            src_arr = torch.tensor(
                [t[0] for t in add_triplets], device=device, dtype=torch.long
            )
            dst_arr = torch.tensor(
                [t[1] for t in add_triplets], device=device, dtype=torch.long
            )
            add_e = torch.stack([src_arr, dst_arr], dim=0)
            edge_corrupt = torch.cat([edge_keep, add_e], dim=1)
            if edge_attr is not None and add_attr_list:
                add_attr_t = torch.stack(add_attr_list, dim=0)
                attr_corrupt = torch.cat([attr_keep, add_attr_t], dim=0)
            else:
                attr_corrupt = attr_keep
        else:
            edge_corrupt = edge_keep
            attr_corrupt = attr_keep

        pairs = None
        if perturb_len_total > 0:
            pairs = {}
            for k in ("b", "i", "j", "y", "target_type"):
                vlist = perturb_tensors.get(k)
                if vlist:
                    pairs[k] = torch.cat(vlist, dim=0)
        return edge_corrupt, attr_corrupt, pairs

    def _sample_hard_negatives_spd(
        self, edge_keep, node_batch, g, nodes, forbidden, max_spd
    ):
        """Sample negative pairs based on shortest-path distance (SPD).

        Uses multi-source BFS for O(V+E) memory instead of dense adjacency matrix.
        Finds node pairs with 2 <= SPD <= max_spd that are not actual edges.
        """
        n = nodes.numel()
        mask = node_batch[edge_keep[0]] == g
        eids = mask.nonzero(as_tuple=False).flatten()
        if eids.numel() == 0:
            return self._sample_random_negatives(nodes, forbidden, 10)

        src_global = edge_keep[0, eids]
        dst_global = edge_keep[1, eids]

        max_node_id = max(
            nodes.max().item(), src_global.max().item(), dst_global.max().item()
        )
        local_idx = torch.full(
            (max_node_id + 1,), -1, dtype=torch.long, device=nodes.device
        )
        local_idx[nodes] = torch.arange(n, device=nodes.device)
        src_local = local_idx[src_global]
        dst_local = local_idx[dst_global]
        valid = (src_local >= 0) & (dst_local >= 0)
        src_local = src_local[valid]
        dst_local = dst_local[valid]

        if src_local.numel() == 0:
            return self._sample_random_negatives(nodes, forbidden, 10)

        adj_list = [[] for _ in range(n)]
        src_cpu = src_local.cpu().tolist()
        dst_cpu = dst_local.cpu().tolist()
        for s, d in zip(src_cpu, dst_cpu):
            adj_list[s].append(d)
            adj_list[d].append(s)

        edge_set = set()
        for s, d in zip(src_cpu, dst_cpu):
            edge_set.add((min(s, d), max(s, d)))

        candidates = []
        node_list = nodes.cpu().tolist()
        for start in range(n):
            if len(candidates) >= 500:
                break
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_spd:
                    continue
                for v in adj_list[u]:
                    if dist[v] != -1:
                        continue
                    dist[v] = dist[u] + 1
                    if dist[v] >= 2:
                        lo, hi = min(start, v), max(start, v)
                        if (lo, hi) not in edge_set:
                            s_g = node_list[lo]
                            d_g = node_list[hi]
                            key = (s_g, d_g)
                            if key not in forbidden:
                                candidates.append(key)
                                forbidden.add(key)
                                if len(candidates) >= 500:
                                    break
                    queue.append(v)

        if not candidates:
            return self._sample_random_negatives(nodes, forbidden, 10)

        random.shuffle(candidates)
        return candidates

    def _sample_random_negatives(self, nodes, forbidden, target):
        """Fallback: random negative sampling."""
        candidates = []
        n = nodes.numel()
        attempts = 0
        max_attempts = target * 20
        while len(candidates) < target and attempts < max_attempts:
            attempts += 1
            i = torch.randint(0, n, (1,)).item()
            j = torch.randint(0, n, (1,)).item()
            if i == j:
                continue
            si, dj = int(nodes[i].item()), int(nodes[j].item())
            key = (si, dj) if si <= dj else (dj, si)
            if key not in forbidden:
                forbidden.add(key)
                candidates.append((si, dj))
        return candidates

    def _compute_pretrain_losses(
        self,
        batch,
        h_out,
        z_out,
        node_mask,
        transport,
        cost,
        atom_mask,
        motif_block_mask,
        motif_id,
        true_adj_dense,
        perturbed_pairs,
        edge_attr=None,
    ):
        losses = {}
        device = h_out.device

        raw_x = getattr(batch, "x_raw", None)
        if raw_x is None:
            losses["mask_atom"] = torch.tensor(0.0, device=device)
        else:
            target_atom = _infer_atom_targets(raw_x).to(device)
            atom_logits = self.atom_decoder(h_out)
            if atom_mask.any():
                losses["mask_atom"] = self.ce_loss(
                    atom_logits[atom_mask], target_atom[atom_mask]
                )
            else:
                losses["mask_atom"] = torch.tensor(0.0, device=device)

        motif_logits = self.motif_decoder(h_out)
        if motif_block_mask.any():
            losses["motif_mask"] = self.ce_loss(
                motif_logits[motif_block_mask], motif_id[motif_block_mask]
            )
        else:
            losses["motif_mask"] = torch.tensor(0.0, device=device)

        edge_pred_dense = self.edge_decoder(z_out)
        # to_dense_adj with undirected edges already produces a symmetric matrix;
        # no need to re-symmetrize.
        true_adj_dense = true_adj_dense.clamp(min=0.0, max=1.0)

        denoise_mode = getattr(cfg.otformer.pretrain, "edge_denoise_mode", "random")

        if perturbed_pairs is not None and perturbed_pairs["y"].numel() > 0:
            b_idx = perturbed_pairs["b"]
            i_idx = perturbed_pairs["i"]
            j_idx = perturbed_pairs["j"]
            labels = perturbed_pairs["y"]
            pred = edge_pred_dense[b_idx, i_idx, j_idx]

            if denoise_mode == "reconstruct" and "target_type" in perturbed_pairs:
                # Reconstruct original edge types for masked edges (y==1).
                mask_pos = labels == 1
                if mask_pos.any():
                    losses["edge_denoise"] = self.ce_loss(
                        pred[mask_pos], perturbed_pairs["target_type"][mask_pos]
                    )
                else:
                    losses["edge_denoise"] = torch.tensor(0.0, device=device)
            elif denoise_mode == "edge_type" and "target_type" in perturbed_pairs:
                losses["edge_denoise"] = self.ce_loss(
                    pred, perturbed_pairs["target_type"]
                )
            else:
                true_edge = true_adj_dense[b_idx, i_idx, j_idx]
                losses["edge_denoise"] = self.bce_loss(pred.squeeze(-1), true_edge)
        else:
            pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            non_diag = ~torch.eye(
                pair_mask.shape[-1], dtype=torch.bool, device=pair_mask.device
            ).unsqueeze(0)
            pair_mask = pair_mask & non_diag
            valid_idx = pair_mask.nonzero(as_tuple=False)
            if valid_idx.shape[0] > 0:
                if denoise_mode == "reconstruct":
                    losses["edge_denoise"] = torch.tensor(0.0, device=device)
                elif denoise_mode == "edge_type":
                    losses["edge_denoise"] = torch.tensor(0.0, device=device)
                else:
                    num_samples = max(
                        1,
                        int(
                            valid_idx.shape[0] * cfg.otformer.pretrain.edge_sample_ratio
                        ),
                    )
                    perm = torch.randperm(valid_idx.shape[0], device=device)[
                        :num_samples
                    ]
                    sampled = valid_idx[perm]
                    pred_edge = edge_pred_dense[
                        sampled[:, 0], sampled[:, 1], sampled[:, 2]
                    ]
                    true_edge = true_adj_dense[
                        sampled[:, 0], sampled[:, 1], sampled[:, 2]
                    ].float()
                    losses["edge_denoise"] = self.bce_loss(pred_edge, true_edge)
            else:
                losses["edge_denoise"] = torch.tensor(0.0, device=device)

        if transport is None or cost is None:
            losses["ot_prior"] = torch.tensor(0.0, device=device)
        else:
            losses["ot_prior"] = (transport * cost).sum(dim=(1, 2)).mean()
        return losses

    def _losses_by_mode(self, losses):
        mode = cfg.otformer.pretrain.mode
        if mode == "joint":
            return losses
        if mode == "atom_only":
            return {
                "mask_atom": losses["mask_atom"],
                "motif_mask": torch.zeros_like(losses["motif_mask"]),
                "edge_denoise": torch.zeros_like(losses["edge_denoise"]),
                "ot_prior": torch.zeros_like(losses["ot_prior"]),
            }
        if mode == "motif_only":
            return {
                "mask_atom": torch.zeros_like(losses["mask_atom"]),
                "motif_mask": losses["motif_mask"],
                "edge_denoise": torch.zeros_like(losses["edge_denoise"]),
                "ot_prior": torch.zeros_like(losses["ot_prior"]),
            }
        if mode == "edge_only":
            return {
                "mask_atom": torch.zeros_like(losses["mask_atom"]),
                "motif_mask": torch.zeros_like(losses["motif_mask"]),
                "edge_denoise": losses["edge_denoise"],
                "ot_prior": torch.zeros_like(losses["ot_prior"]),
            }
        if mode == "no_ot":
            return {
                "mask_atom": losses["mask_atom"],
                "motif_mask": losses["motif_mask"],
                "edge_denoise": losses["edge_denoise"],
                "ot_prior": torch.zeros_like(losses["ot_prior"]),
            }
        raise ValueError(
            f"Unsupported otformer.pretrain.mode='{mode}'. "
            "Choose one of: joint, atom_only, motif_only, edge_only, no_ot."
        )

    def forward(self, batch):
        batch.x_raw = batch.x.clone()
        batch.edge_attr_raw = (
            batch.edge_attr.clone()
            if hasattr(batch, "edge_attr") and batch.edge_attr is not None
            else None
        )
        batch = self.encoder(batch)
        if hasattr(self, "pre_mp"):
            batch = self.pre_mp(batch)

        # Preserve clean graph for denoising targets.
        edge_index_true = batch.edge_index
        edge_attr_true = getattr(batch, "edge_attr", None)
        edge_attr_raw = getattr(batch, "edge_attr_raw", None)
        node_batch = batch.batch
        h_encoded = batch.x

        edge_type_target = _infer_edge_type_targets(edge_attr_raw)

        if self.use_rrwp and (
            (not hasattr(batch, f"{self.rrwp_attr_name}_index"))
            or (not hasattr(batch, f"{self.rrwp_attr_name}_val"))
        ):
            raise ValueError(
                "RRWP pair bias is enabled but precomputed RRWP stats are missing. "
                f"Expected '{self.rrwp_attr_name}_index' and '{self.rrwp_attr_name}_val'. "
                "Enable posenc_RRWP in config to precompute RRWP."
            )

        # Optional input edge perturbation for denoising pretraining.
        perturb_pairs = None
        edge_index_work = edge_index_true
        edge_attr_work = edge_attr_true
        pretrain_on = getattr(cfg.otformer.pretrain, "enable", False) and self.training
        denoise_mode = getattr(cfg.otformer.pretrain, "edge_denoise_mode", "random")
        if (
            pretrain_on
            and denoise_mode in {"reconstruct", "edge_type"}
            and edge_type_target is None
        ):
            raise ValueError(
                f"edge_denoise_mode='{denoise_mode}' requires raw edge types in batch.edge_attr"
            )
        if pretrain_on:
            edge_index_work, edge_attr_work, perturb_pairs = self._perturb_edges(
                edge_index_true,
                edge_attr_true,
                node_batch,
                edge_type_target=edge_type_target,
            )

        # Use perturbed graph for path extraction and OT matching.
        if pretrain_on:
            batch_work = batch.clone()
            batch_work.edge_index = edge_index_work
            if edge_attr_true is not None:
                batch_work.edge_attr = edge_attr_work
        else:
            batch_work = batch

        if self.disable_rum_ot:
            path_repr = None
            node_motif = None
            transport = None
            cost = None
            motif_id = None
        else:
            path_repr, _ = self.rum(
                batch_work, h_encoded, e=getattr(batch_work, "edge_attr", None)
            )
            if path_repr.dim() != 3:
                raise RuntimeError(
                    f"Expected 3D path tensor from RUM, got shape {path_repr.shape}"
                )
            # RUM returns [W, N, D]
            path_repr = path_repr.transpose(0, 1).contiguous()

            node_motif, transport, cost = self.ot_memory(
                path_repr,
                sinkhorn_eps=cfg.otformer.motif.sinkhorn_eps,
                sinkhorn_iters=cfg.otformer.motif.sinkhorn_iters,
                log_domain=cfg.otformer.motif.log_domain,
            )
            motif_id = transport.mean(dim=1).argmax(dim=-1)

        # Build pretraining masks: atom random mask + motif-connected block mask.
        atom_mask = torch.zeros(
            h_encoded.size(0), dtype=torch.bool, device=h_encoded.device
        )
        motif_block_mask = torch.zeros_like(atom_mask)
        mask_union = atom_mask | motif_block_mask
        h_input = h_encoded
        if pretrain_on and not self.disable_rum_ot:
            atom_mask = (
                torch.rand(h_encoded.shape[0], device=h_encoded.device)
                < cfg.otformer.pretrain.atom_mask_ratio
            )
            motif_block_mask = self._sample_motif_block_mask(
                node_batch=node_batch,
                edge_index=edge_index_true,
                motif_id=motif_id,
            )
            mask_union = atom_mask | motif_block_mask
            h_input = h_encoded.clone()
            h_input[mask_union] = self.mask_token.to(h_input.dtype)
        elif pretrain_on and self.disable_rum_ot:
            atom_mask = (
                torch.rand(h_encoded.shape[0], device=h_encoded.device)
                < cfg.otformer.pretrain.atom_mask_ratio
            )
            mask_union = atom_mask
            h_input = h_encoded.clone()
            h_input[mask_union] = self.mask_token.to(h_input.dtype)

        if self.disable_rum_ot:
            h0 = h_input
        else:
            path_mean = path_repr.mean(dim=1)
            if pretrain_on:
                path_mean = path_mean.clone()
                path_mean[mask_union] = self.mask_token.to(path_mean.dtype)
                path_mean = path_mean.detach()

            h0 = h_input + self.motif_to_node(node_motif) + self.path_to_node(path_mean)

        batch_work.x = h0
        z0, h_dense0, node_mask, rrwp_dense = build_pair_init(
            batch_work,
            self.dim_h,
            self.edge_proj,
            self.pair_init_proj,
            use_spd=self.use_spd,
            spd_max_dist=self.spd_max_dist,
            use_rrwp=self.use_rrwp,
            rrwp_attr_name=self.rrwp_attr_name,
            rrwp_dim=self.rrwp_dim,
        )
        h, z = h_dense0, z0
        for t in range(self.recycling_iters):
            if t > 0:
                h_prev = self.h_recycle_norm(h)
                z_prev = self.z_recycle_norm(z)
                if self.detach_recycle:
                    h_prev = h_prev.detach()
                    z_prev = z_prev.detach()
                h = h_prev + h_dense0
                z = z_prev + z0
            for block in self.blocks:
                h, z = block(h, z, node_mask, rrwp_dense=rrwp_dense)

        h_out = h[node_mask]
        batch.x = h_out
        batch.otformer_aux = {
            "transport": transport,
            "cost": cost,
            "atom_mask": atom_mask,
            "motif_block_mask": motif_block_mask,
            "z_out": z,
            "node_mask": node_mask,
            "disable_rum_ot": self.disable_rum_ot,
        }
        if getattr(cfg.otformer.pretrain, "enable", False):
            true_adj = to_dense_adj(
                edge_index_true,
                batch=node_batch,
                max_num_nodes=node_mask.shape[1],
            ).to(h_out.device)
            losses = self._compute_pretrain_losses(
                batch=batch,
                h_out=h_out,
                z_out=z,
                node_mask=node_mask,
                transport=transport,
                cost=cost,
                atom_mask=atom_mask,
                motif_block_mask=motif_block_mask,
                motif_id=motif_id,
                true_adj_dense=true_adj,
                perturbed_pairs=perturb_pairs,
                edge_attr=edge_attr_true,
            )
            batch.otformer_aux["losses_raw"] = losses
            batch.otformer_aux["losses"] = self._losses_by_mode(losses)

        return self.post_mp(batch)
