from typing import Optional, Tuple

import torch

try:
    from torch_cluster import random_walk as pyg_random_walk
except Exception:  # pragma: no cover - optional dependency at runtime
    pyg_random_walk = None


def _get_edge_index_and_num_nodes(
    g: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    if hasattr(g, "edge_index"):
        edge_index = g.edge_index
        num_nodes = g.num_nodes
    else:
        edge_index = g
        num_nodes = None
    if num_nodes is None:
        if edge_index.numel() == 0:
            num_nodes = 0
        else:
            num_nodes = int(edge_index.max().item()) + 1
    return edge_index, num_nodes


def uniform_random_walk(
    g: torch.Tensor,
    num_samples: int,
    length: int,
    subsample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random walk on a graph.

    Parameters
    ----------
    g : torch_geometric.data.Data or edge_index Tensor
        The graph.
    num_samples : int
        Number of random walks per node.
    length : int
        Length of each random walk.

    Returns
    -------
    walks : Tensor
        The random walks.
    eids : Tensor
        Edge indices for each step in the walk.
    """
    if length < 1:
        raise ValueError(f"`length` must be >= 1, got {length}.")

    edge_index, num_nodes = _get_edge_index_and_num_nodes(g)
    if subsample is None:
        nodes = torch.arange(
            num_nodes,
            device=edge_index.device,
            dtype=edge_index.dtype,
        )
        nodes = nodes.repeat(num_samples)
        num_walk_nodes = num_nodes
    else:
        nodes = subsample.to(
            device=edge_index.device,
            dtype=edge_index.dtype,
        ).repeat(num_samples)
        num_walk_nodes = subsample.size(0)
    if nodes.numel() > 0:
        invalid_nodes = (nodes < 0) | (nodes >= num_nodes)
        if invalid_nodes.any():
            bad_node = int(nodes[invalid_nodes][0].item())
            raise ValueError(
                f"`subsample` contains invalid node id {bad_node}. "
                f"Valid range is [0, {num_nodes - 1}]."
            )

    if pyg_random_walk is None:
        raise ImportError(
            "torch_cluster is required for random walks. "
            "Please install torch_cluster."
        )

    walks, eids = pyg_random_walk(
        edge_index[0],
        edge_index[1],
        nodes,
        length - 1,
        coalesced=False,
        num_nodes=num_nodes,
        return_edge_indices=True,
    )
    walks = walks.view(num_samples, num_walk_nodes, length)
    eids = eids.view(num_samples, num_walk_nodes, length - 1)
    return walks, eids


# @torch.jit.trace(example_inputs=(torch.zeros(10, 10, 10)))


def uniqueness(walk):
    """
    Compute the uniqueness of a random walk.

    Parameters
    ----------
    walk : Tensor
        The random walk.

    Returns
    -------
    uniqueness : Tensor
        The uniqueness of the random walk.
    """
    walk_equal = walk.unsqueeze(-1) == walk.unsqueeze(-2)
    walk_equal = (1 * walk_equal).argmax(dim=-1)
    return walk_equal
