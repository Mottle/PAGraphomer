import torch
from torch_cluster import random_walk as cluster_random_walk
from torch_geometric.data import Data

from rum.random_walk import uniform_random_walk


def test_shape():
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    g = Data(edge_index=edge_index, num_nodes=6)
    walks, eids = uniform_random_walk(g, 2, 3)
    assert walks.shape == (2, 6, 3)
    assert eids.shape == (2, 6, 2)


def test_shape_long_walk():
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    g = Data(edge_index=edge_index, num_nodes=6)
    walks, eids = uniform_random_walk(g, 2, 5)
    assert walks.shape == (2, 6, 5)
    assert eids.shape == (2, 6, 4)


def test_shape_with_subsample():
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    g = Data(edge_index=edge_index, num_nodes=6)
    subsample = torch.tensor([0, 2, 4], dtype=torch.int32)
    walks, eids = uniform_random_walk(g, 3, 4, subsample=subsample)
    assert walks.shape == (3, 3, 4)
    assert eids.shape == (3, 3, 3)


def test_multigraph_edge_ids_match_cluster():
    edge_index = torch.tensor(
        [[0, 0, 0, 1, 1, 2, 2, 2], [1, 1, 2, 2, 2, 0, 0, 1]],
        dtype=torch.long,
    )
    g = Data(edge_index=edge_index, num_nodes=3)
    nodes = torch.arange(3, dtype=torch.long).repeat(2)

    torch.manual_seed(7)
    expected_walks, expected_eids = cluster_random_walk(
        edge_index[0],
        edge_index[1],
        nodes,
        4,
        coalesced=False,
        num_nodes=3,
        return_edge_indices=True,
    )
    expected_walks = expected_walks.view(2, 3, 5)
    expected_eids = expected_eids.view(2, 3, 4)

    torch.manual_seed(7)
    walks, eids = uniform_random_walk(g, 2, 5)
    torch.testing.assert_close(walks, expected_walks)
    torch.testing.assert_close(eids, expected_eids)


def test_invalid_subsample_raises():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    g = Data(edge_index=edge_index, num_nodes=2)
    subsample = torch.tensor([0, 99], dtype=torch.long)
    try:
        uniform_random_walk(g, 1, 3, subsample=subsample)
    except ValueError as exc:
        assert "invalid node id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid subsample.")


def test_invalid_length_raises():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    g = Data(edge_index=edge_index, num_nodes=2)
    try:
        uniform_random_walk(g, 1, 0)
    except ValueError as exc:
        assert "must be >= 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid length.")
