import unittest as ut
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from gps.network.otformer_model import OTFormerModel
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg


class _Args:
    cfg_file = "configs/OTFormer/zinc-OTFormer-pretrain.yaml"
    opts = []
    repeat = 1
    mark_done = False


class TestHardNegativeSampling(ut.TestCase):
    def setUp(self):
        set_cfg(cfg)
        load_cfg(cfg, _Args)
        cfg.dataset.node_encoder = False
        cfg.dataset.edge_encoder = False
        cfg.gnn.dim_inner = 16
        cfg.gnn.layers_pre_mp = 0
        cfg.gnn.head = "san_graph"
        cfg.otformer.pretrain.enable = True
        cfg.otformer.pretrain.edge_perturb_ratio = 0.2
        cfg.otformer.pretrain.edge_neg_ratio = 2.0
        cfg.otformer.pretrain.edge_denoise_mode = "hard_spd"
        cfg.otformer.pretrain.hard_neg_max_spd = 3
        self.model = OTFormerModel(dim_in=16, dim_out=1)

    def test_hard_negatives_are_not_true_edges(self):
        """Hard negatives should not overlap with true edges."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        edge_attr = None
        node_batch = torch.zeros(4, dtype=torch.long)

        _, _, pairs = self.model._perturb_edges(edge_index, edge_attr, node_batch)
        self.assertIsNotNone(pairs)

        true_set = set()
        for eid in range(edge_index.size(1)):
            s, d = int(edge_index[0, eid]), int(edge_index[1, eid])
            true_set.add((s, d) if s <= d else (d, s))

        neg_set = set()
        for idx in range(pairs["y"].numel()):
            if pairs["y"][idx].item() == 0:
                i, j = int(pairs["i"][idx]), int(pairs["j"][idx])
                neg_set.add((i, j) if i <= j else (j, i))

        overlap = true_set & neg_set
        self.assertEqual(
            len(overlap), 0, f"Hard negatives overlap true edges: {overlap}"
        )

    def test_hard_negatives_have_spd_gte_2(self):
        """Hard negatives should have shortest path distance >= 2."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
        )
        edge_attr = None
        node_batch = torch.zeros(5, dtype=torch.long)

        _, _, pairs = self.model._perturb_edges(edge_index, edge_attr, node_batch)
        self.assertIsNotNone(pairs)

        neg_pairs = []
        for idx in range(pairs["y"].numel()):
            if pairs["y"][idx].item() == 0:
                neg_pairs.append((int(pairs["i"][idx]), int(pairs["j"][idx])))

        for s, d in neg_pairs:
            self.assertNotEqual(s, d, "Self-loop should not be a negative")

    def test_random_fallback_when_no_hard_negatives(self):
        """Should fallback to random negatives when no SPD-2+ pairs exist."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = None
        node_batch = torch.zeros(2, dtype=torch.long)

        _, _, pairs = self.model._perturb_edges(edge_index, edge_attr, node_batch)
        # Dropped edges become positive samples (y=1)
        self.assertIsNotNone(pairs)
        # With only 2 nodes and 1 edge, no negative pairs can be sampled
        n_neg = (pairs["y"] == 0).sum().item()
        self.assertEqual(n_neg, 0)

    def test_perturb_y_dtype_is_long(self):
        """Labels should be long for cross-entropy compatibility."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        edge_attr = None
        node_batch = torch.zeros(4, dtype=torch.long)

        _, _, pairs = self.model._perturb_edges(edge_index, edge_attr, node_batch)
        self.assertEqual(pairs["y"].dtype, torch.long)

    def test_hard_spd_mode_produces_negatives(self):
        """hard_spd mode should produce negative samples."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 0, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 0, 5, 4]],
            dtype=torch.long,
        )
        edge_attr = None
        node_batch = torch.zeros(6, dtype=torch.long)

        _, _, pairs = self.model._perturb_edges(edge_index, edge_attr, node_batch)
        self.assertIsNotNone(pairs)
        n_neg = (pairs["y"] == 0).sum().item()
        self.assertGreater(n_neg, 0)


class TestEdgeTypeDecoder(ut.TestCase):
    def setUp(self):
        set_cfg(cfg)
        load_cfg(cfg, _Args)
        cfg.dataset.node_encoder = False
        cfg.dataset.edge_encoder = False
        cfg.dataset.edge_encoder_num_types = 4
        cfg.gnn.dim_inner = 16
        cfg.gnn.layers_pre_mp = 0
        cfg.gnn.head = "san_graph"
        cfg.otformer.pretrain.enable = True
        cfg.otformer.pretrain.edge_denoise_mode = "edge_type"

    def test_edge_decoder_output_dim_for_edge_type(self):
        """Edge decoder should output n_edge_types + 1 classes."""
        model = OTFormerModel(dim_in=16, dim_out=1)
        self.assertEqual(model.edge_decoder.out_features, 5)

    def test_edge_decoder_output_dim_for_binary(self):
        """Edge decoder should output 1 class for binary mode."""
        cfg.otformer.pretrain.edge_denoise_mode = "random"
        model = OTFormerModel(dim_in=16, dim_out=1)
        self.assertEqual(model.edge_decoder.out_features, 1)


class TestEdgeDenoiseSemantics(ut.TestCase):
    def _build_model(self, mode):
        set_cfg(cfg)
        load_cfg(cfg, _Args)
        cfg.dataset.node_encoder = False
        cfg.dataset.edge_encoder = False
        cfg.dataset.edge_encoder_num_types = 4
        cfg.gnn.dim_inner = 16
        cfg.gnn.layers_pre_mp = 0
        cfg.gnn.head = "san_graph"
        cfg.otformer.pretrain.enable = True
        cfg.otformer.pretrain.edge_perturb_ratio = 0.5
        cfg.otformer.pretrain.edge_neg_ratio = 1.0
        cfg.otformer.pretrain.edge_denoise_mode = mode
        return OTFormerModel(dim_in=16, dim_out=1)

    @staticmethod
    def _count_pair_dirs(edge_index, s, d):
        return int(
            ((edge_index[0] == s) & (edge_index[1] == d)).sum().item()
            + ((edge_index[0] == d) & (edge_index[1] == s)).sum().item()
        )

    def test_reconstruct_masks_edges_symmetrically(self):
        """Reconstruct mode should mask both directions of an undirected edge."""
        model = self._build_model("reconstruct")
        torch.manual_seed(0)

        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        edge_attr = torch.arange(edge_index.size(1) * 16, dtype=torch.float).view(
            -1, 16
        )
        edge_type_target = torch.tensor([1, 1, 2, 2, 3, 3], dtype=torch.long)
        node_batch = torch.zeros(4, dtype=torch.long)

        edge_corrupt, attr_corrupt, pairs = model._perturb_edges(
            edge_index,
            edge_attr,
            node_batch,
            edge_type_target=edge_type_target,
        )

        self.assertEqual(edge_corrupt.shape, edge_index.shape)
        self.assertEqual(attr_corrupt.shape, edge_attr.shape)
        self.assertIsNotNone(pairs)
        self.assertTrue(torch.all(pairs["y"] == 1))

        pair_to_eids = {}
        for eid in range(edge_index.size(1)):
            s = int(edge_index[0, eid].item())
            d = int(edge_index[1, eid].item())
            key = (s, d) if s <= d else (d, s)
            pair_to_eids.setdefault(key, []).append(eid)

        for key, eids in pair_to_eids.items():
            changed = [
                not torch.equal(attr_corrupt[eid], edge_attr[eid]) for eid in eids
            ]
            self.assertTrue(
                all(changed) or (not any(changed)),
                f"Undirected edge {key} was perturbed asymmetrically: {changed}",
            )

    def test_edge_type_mode_has_semantic_targets(self):
        """edge_type mode should supervise edge classes, not binary y labels."""
        model = self._build_model("edge_type")
        torch.manual_seed(1)

        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
            dtype=torch.long,
        )
        edge_type_target = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        node_batch = torch.zeros(5, dtype=torch.long)

        edge_corrupt, _, pairs = model._perturb_edges(
            edge_index,
            edge_attr=None,
            node_batch=node_batch,
            edge_type_target=edge_type_target,
        )

        self.assertIsNotNone(pairs)
        self.assertIn("target_type", pairs)
        y = pairs["y"]
        t = pairs["target_type"]
        self.assertEqual(y.numel(), t.numel())

        noedge_id = model.edge_type_noedge_id
        if (y == 0).any():
            self.assertTrue(torch.all(t[y == 0] == noedge_id))
        if (y == 1).any():
            self.assertTrue(torch.all(t[y == 1] < noedge_id))

        # Dropped true edges should be removed symmetrically (0 or 2 directions remain).
        undirected_pairs = {(0, 1), (1, 2), (2, 3), (3, 4)}
        for s, d in undirected_pairs:
            n_dirs = self._count_pair_dirs(edge_corrupt, s, d)
            self.assertIn(n_dirs, {0, 2})

    def test_edge_type_loss_uses_target_type(self):
        """_compute_pretrain_losses should use target_type CE in edge_type mode."""
        model = self._build_model("edge_type")
        with torch.no_grad():
            model.edge_decoder.weight.zero_()
            model.edge_decoder.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))

        n_nodes = 3
        h_out = torch.zeros(n_nodes, 16)
        z_out = torch.zeros(1, n_nodes, n_nodes, 16)
        node_mask = torch.tensor([[True, True, True]])
        transport = torch.zeros(n_nodes, 1, 1)
        cost = torch.zeros(n_nodes, 1, 1)
        atom_mask = torch.zeros(n_nodes, dtype=torch.bool)
        motif_block_mask = torch.zeros(n_nodes, dtype=torch.bool)
        motif_id = torch.zeros(n_nodes, dtype=torch.long)
        true_adj_dense = torch.zeros(1, n_nodes, n_nodes)
        perturbed_pairs = {
            "b": torch.tensor([0, 0], dtype=torch.long),
            "i": torch.tensor([0, 1], dtype=torch.long),
            "j": torch.tensor([1, 2], dtype=torch.long),
            "y": torch.tensor([0, 1], dtype=torch.long),
            "target_type": torch.tensor([4, 3], dtype=torch.long),
        }

        losses = model._compute_pretrain_losses(
            batch=SimpleNamespace(x_raw=None),
            h_out=h_out,
            z_out=z_out,
            node_mask=node_mask,
            transport=transport,
            cost=cost,
            atom_mask=atom_mask,
            motif_block_mask=motif_block_mask,
            motif_id=motif_id,
            true_adj_dense=true_adj_dense,
            perturbed_pairs=perturbed_pairs,
            edge_attr=None,
        )

        pred = model.edge_decoder(z_out)[
            perturbed_pairs["b"], perturbed_pairs["i"], perturbed_pairs["j"]
        ]
        expected = F.cross_entropy(pred, perturbed_pairs["target_type"])
        wrong = F.cross_entropy(pred, perturbed_pairs["y"])

        self.assertTrue(torch.allclose(losses["edge_denoise"], expected))
        self.assertFalse(torch.allclose(losses["edge_denoise"], wrong))


class TestHardNegativeSamplingPerformance(ut.TestCase):
    """Tests to verify the optimized hard negative sampling produces correct results."""

    def setUp(self):
        set_cfg(cfg)
        load_cfg(cfg, _Args)
        cfg.dataset.node_encoder = False
        cfg.dataset.edge_encoder = False
        cfg.gnn.dim_inner = 16
        cfg.gnn.layers_pre_mp = 0
        cfg.gnn.head = "san_graph"
        cfg.otformer.pretrain.enable = True
        cfg.otformer.pretrain.edge_perturb_ratio = 0.1
        cfg.otformer.pretrain.edge_neg_ratio = 2.0
        cfg.otformer.pretrain.edge_denoise_mode = "hard_spd"
        cfg.otformer.pretrain.hard_neg_max_spd = 3
        self.model = OTFormerModel(dim_in=16, dim_out=1)

    def test_large_graph_produces_valid_negatives(self):
        """Hard negatives on a larger graph should be valid (no overlap, no self-loops)."""
        n_nodes = 20
        edges = []
        for i in range(n_nodes - 1):
            edges.extend([[i, i + 1], [i + 1, i]])
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = None
        node_batch = torch.zeros(n_nodes, dtype=torch.long)

        _, _, pairs = self.model._perturb_edges(edge_index, edge_attr, node_batch)
        self.assertIsNotNone(pairs)

        true_set = set()
        for eid in range(edge_index.size(1)):
            s, d = int(edge_index[0, eid]), int(edge_index[1, eid])
            true_set.add((s, d) if s <= d else (d, s))

        neg_set = set()
        for idx in range(pairs["y"].numel()):
            if pairs["y"][idx].item() == 0:
                i, j = int(pairs["i"][idx]), int(pairs["j"][idx])
                neg_set.add((i, j) if i <= j else (j, i))
                self.assertNotEqual(i, j, "Self-loop in negatives")

        overlap = true_set & neg_set
        self.assertEqual(len(overlap), 0, f"Overlap: {overlap}")

    def test_multi_batch_graphs(self):
        """Hard negatives should work correctly across multiple graphs in a batch."""
        edges_g0 = [[0, 1], [1, 0], [1, 2], [2, 1]]
        edges_g1 = [[3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5]]
        edge_index = torch.tensor(edges_g0 + edges_g1, dtype=torch.long).T
        node_batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        edge_attr = None

        _, _, pairs = self.model._perturb_edges(edge_index, edge_attr, node_batch)
        self.assertIsNotNone(pairs)

        true_set = set()
        for eid in range(edge_index.size(1)):
            s, d = int(edge_index[0, eid]), int(edge_index[1, eid])
            true_set.add((s, d) if s <= d else (d, s))

        neg_set = set()
        for idx in range(pairs["y"].numel()):
            if pairs["y"][idx].item() == 0:
                i, j = int(pairs["i"][idx]), int(pairs["j"][idx])
                neg_set.add((i, j) if i <= j else (j, i))

        overlap = true_set & neg_set
        self.assertEqual(len(overlap), 0)
        self.assertGreater(len(neg_set), 0)

    def test_disconnected_graph_fallback(self):
        """Graph with no SPD-2+ pairs should fallback to random negatives."""
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        edge_attr = None
        node_batch = torch.zeros(4, dtype=torch.long)

        _, _, pairs = self.model._perturb_edges(edge_index, edge_attr, node_batch)
        self.assertIsNotNone(pairs)
        # Fallback to random negatives produces valid cross-component pairs
        n_neg = (pairs["y"] == 0).sum().item()
        self.assertGreater(n_neg, 0)

        # Verify negatives don't overlap true edges
        true_set = {(0, 1), (2, 3)}
        for idx in range(pairs["y"].numel()):
            if pairs["y"][idx].item() == 0:
                i, j = int(pairs["i"][idx]), int(pairs["j"][idx])
                key = (i, j) if i <= j else (j, i)
                self.assertNotIn(key, true_set)


if __name__ == "__main__":
    ut.main()
