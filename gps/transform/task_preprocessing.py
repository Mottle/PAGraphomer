import networkx as nx
import torch
from torch_geometric.utils import to_networkx


def _add_otformer_spd(data, max_dist):
    """Precompute clipped shortest-path distances for OTFormer pair init."""
    if max_dist <= 0:
        raise ValueError(f"otformer.pair.spd_max_dist must be > 0, got {max_dist}")

    num_nodes = int(data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0))
    if num_nodes == 0:
        data.ot_spd_index = torch.empty((2, 0), dtype=torch.long)
        data.ot_spd_val = torch.empty((0,), dtype=torch.long)
        return data

    graph = to_networkx(data, to_undirected=True)
    shortest_paths = dict(nx.shortest_path_length(graph))

    spd = torch.full((num_nodes, num_nodes), max_dist, dtype=torch.long)
    for i in range(num_nodes):
        spd[i, i] = 0
    for i, dist_map in shortest_paths.items():
        for j, d in dist_map.items():
            spd[int(i), int(j)] = min(int(d), max_dist)

    idx = torch.arange(num_nodes, dtype=torch.long)
    graph_index = torch.stack([idx.repeat_interleave(num_nodes), idx.repeat(num_nodes)])
    data.ot_spd_index = graph_index
    data.ot_spd_val = spd.reshape(-1)
    return data


def shuffle(tensor):
    idx = torch.randperm(len(tensor))
    return tensor[idx]


def task_specific_preprocessing(data, cfg):
    """Task-specific preprocessing before the dataset is logged and finalized.

    Args:
        data: PyG graph
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    if cfg.gnn.head == "infer_links":
        N = data.x.size(0)
        idx = torch.arange(N, dtype=torch.long)
        complete_index = torch.stack([idx.repeat_interleave(N), idx.repeat(N)], 0)

        data.edge_attr = None

        if cfg.dataset.infer_link_label == "edge":
            labels = torch.empty(N, N, dtype=torch.long)
            non_edge_index = (
                (complete_index.T.unsqueeze(1) != data.edge_index.T)
                .any(2)
                .all(1)
                .nonzero()[:, 0]
            )
            non_edge_index = shuffle(non_edge_index)[: data.edge_index.size(1)]
            edge_index = (
                (complete_index.T.unsqueeze(1) == data.edge_index.T)
                .all(2)
                .any(1)
                .nonzero()[:, 0]
            )

            final_index = shuffle(torch.cat([edge_index, non_edge_index]))
            data.complete_edge_index = complete_index[:, final_index]

            labels.fill_(0)
            labels[data.edge_index[0], data.edge_index[1]] = 1

            assert labels.flatten()[final_index].mean(dtype=torch.float) == 0.5
        else:
            raise ValueError(
                f"Infer-link task {cfg.dataset.infer_link_label} not available."
            )

        data.y = labels.flatten()[final_index]

    supported_encoding_available = (
        cfg.posenc_LapPE.enable
        or cfg.posenc_RWSE.enable
        or cfg.posenc_RRWP.enable
        or cfg.posenc_GraphormerBias.enable
    )

    if cfg.dataset.name == "TRIANGLES":

        # If encodings are present they can append to the empty data.x
        if not supported_encoding_available:
            data.x = torch.zeros((data.x.size(0), 1))
        data.y = data.y.sub(1).to(torch.long)

    if cfg.dataset.name == "CSL":

        # If encodings are present they can append to the empty data.x
        if not supported_encoding_available:
            data.x = torch.zeros((data.num_nodes, 1))
        else:
            data.x = torch.zeros((data.num_nodes, 0))

    if getattr(cfg.model, "type", "") == "OTFormerModel" and getattr(
        cfg.otformer.pair, "use_spd", False
    ):
        data = _add_otformer_spd(data, int(cfg.otformer.pair.spd_max_dist))

    return data
