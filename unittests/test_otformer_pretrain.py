import unittest as ut
from types import SimpleNamespace
import tempfile
from pathlib import Path

import torch
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg

import gps  # noqa: F401, register modules
from gps.layer.otformer_layer import sinkhorn_transport
from gps.network.otformer_model import OTFormerModel
from gps.train.otformer_pretrain import _save_epoch_weights_rolling


class _Args:
    cfg_file = "configs/OTFormer/zinc-OTFormer-pretrain.yaml"
    opts = []
    repeat = 1
    mark_done = False


def _local_index(node_batch):
    out = torch.zeros_like(node_batch)
    num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 0
    for g in range(num_graphs):
        nodes = (node_batch == g).nonzero(as_tuple=False).flatten()
        out[nodes] = torch.arange(nodes.numel(), device=node_batch.device)
    return out


class TestOTFormerPretrain(ut.TestCase):
    def setUp(self):
        set_cfg(cfg)
        load_cfg(cfg, _Args)
        cfg.dataset.node_encoder = False
        cfg.dataset.edge_encoder = False
        cfg.gnn.dim_inner = 16
        cfg.gnn.layers_pre_mp = 0
        cfg.otformer.pretrain.enable = True
        cfg.otformer.pretrain.edge_denoise_mode = "random"
        self.model = OTFormerModel(dim_in=16, dim_out=1)

    def test_motif_mask_ratio_zero_disables_block_mask(self):
        cfg.otformer.pretrain.motif_mask_ratio = 0.0
        node_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        edge_index = torch.tensor(
            [[0, 1, 1, 3], [1, 0, 2, 4]],
            dtype=torch.long,
        )
        motif_id = torch.tensor([2, 2, 3, 1, 1], dtype=torch.long)
        mask = self.model._sample_motif_block_mask(
            node_batch=node_batch,
            edge_index=edge_index,
            motif_id=motif_id,
        )
        self.assertFalse(mask.any().item())

    def test_pretrain_mode_gating(self):
        base_losses = {
            "mask_atom": torch.tensor(1.0),
            "motif_mask": torch.tensor(2.0),
            "edge_denoise": torch.tensor(3.0),
            "ot_prior": torch.tensor(4.0),
        }

        cfg.otformer.pretrain.mode = "edge_only"
        got = self.model._losses_by_mode(base_losses)
        self.assertEqual(float(got["mask_atom"]), 0.0)
        self.assertEqual(float(got["motif_mask"]), 0.0)
        self.assertEqual(float(got["ot_prior"]), 0.0)
        self.assertEqual(float(got["edge_denoise"]), 3.0)

        cfg.otformer.pretrain.mode = "no_ot"
        got = self.model._losses_by_mode(base_losses)
        self.assertEqual(float(got["ot_prior"]), 0.0)
        self.assertEqual(float(got["mask_atom"]), 1.0)
        self.assertEqual(float(got["motif_mask"]), 2.0)
        self.assertEqual(float(got["edge_denoise"]), 3.0)

        cfg.otformer.pretrain.mode = "bad_mode"
        with self.assertRaises(ValueError):
            self.model._losses_by_mode(base_losses)

    def test_edge_negatives_do_not_overlap_true_edges(self):
        cfg.otformer.pretrain.edge_perturb_ratio = 0.5
        cfg.otformer.pretrain.edge_neg_ratio = 1.0

        # Two disjoint graphs in one batch.
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 3, 4, 4, 5],
                [1, 0, 2, 1, 4, 3, 5, 4],
            ],
            dtype=torch.long,
        )
        edge_attr = torch.randn(edge_index.shape[1], 16)
        node_batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        _, _, pairs = self.model._perturb_edges(edge_index, edge_attr, node_batch)
        self.assertIsNotNone(pairs)

        local_idx = _local_index(node_batch)
        edge_graph = node_batch[edge_index[0]]
        true_local = {}
        num_graphs = int(node_batch.max().item()) + 1
        for g in range(num_graphs):
            eids = (edge_graph == g).nonzero(as_tuple=False).flatten()
            s = local_idx[edge_index[0, eids]].tolist()
            d = local_idx[edge_index[1, eids]].tolist()
            true_set = set(zip(s, d))
            true_set |= {(b, a) for (a, b) in true_set}
            true_local[g] = true_set

        neg_mask = pairs["y"] == 0
        neg_b = pairs["b"][neg_mask].tolist()
        neg_i = pairs["i"][neg_mask].tolist()
        neg_j = pairs["j"][neg_mask].tolist()
        for b, i, j in zip(neg_b, neg_i, neg_j):
            self.assertNotIn((i, j), true_local[b])

    def test_edge_perturb_pairs_are_bidirectional(self):
        cfg.otformer.pretrain.edge_perturb_ratio = 0.5
        cfg.otformer.pretrain.edge_neg_ratio = 1.0

        edge_index = torch.tensor(
            [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]],
            dtype=torch.long,
        )
        edge_attr = torch.randn(edge_index.shape[1], 16)
        node_batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        _, _, pairs = self.model._perturb_edges(edge_index, edge_attr, node_batch)
        self.assertIsNotNone(pairs)

        for label in [0.0, 1.0]:
            mask = pairs["y"] == label
            triples = list(
                zip(
                    pairs["b"][mask].tolist(),
                    pairs["i"][mask].tolist(),
                    pairs["j"][mask].tolist(),
                )
            )
            triple_set = set(triples)
            for b, i, j in triples:
                self.assertIn((b, j, i), triple_set)

    def test_motif_mask_is_component_block(self):
        cfg.otformer.pretrain.motif_mask_ratio = 0.5
        cfg.otformer.pretrain.motif_topk = 1
        node_batch = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]],
            dtype=torch.long,
        )
        motif_id = torch.tensor([7, 7, 7, 7, 1, 1], dtype=torch.long)

        torch.manual_seed(0)
        for _ in range(20):
            mask = self.model._sample_motif_block_mask(
                node_batch=node_batch,
                edge_index=edge_index,
                motif_id=motif_id,
            )
            # Two disconnected components for motif 7 are {0,1} and {2,3};
            # each component must be masked as a whole.
            self.assertTrue(mask[0].item() == mask[1].item())
            self.assertTrue(mask[2].item() == mask[3].item())

    def test_ot_prior_uses_sum_over_path_and_memory(self):
        d = self.model.dim_h
        batch = SimpleNamespace()
        batch.x_raw = None

        h_out = torch.zeros(2, d)
        z_out = torch.randn(1, 2, 2, d)
        node_mask = torch.tensor([[True, True]])
        transport = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.2, 0.1], [0.4, 0.3]],
            ],
            dtype=torch.float,
        )
        cost = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [4.0, 3.0]],
            ],
            dtype=torch.float,
        )
        atom_mask = torch.tensor([False, False])
        motif_block_mask = torch.tensor([False, False])
        motif_id = torch.tensor([0, 0], dtype=torch.long)
        true_adj_dense = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float)
        perturbed_pairs = {
            "b": torch.tensor([0], dtype=torch.long),
            "i": torch.tensor([0], dtype=torch.long),
            "j": torch.tensor([1], dtype=torch.long),
            "y": torch.tensor([0.0]),
        }

        losses = self.model._compute_pretrain_losses(
            batch=batch,
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
        )
        expected = (transport * cost).sum(dim=(1, 2)).mean()
        torch.testing.assert_close(losses["ot_prior"], expected)

    def test_edge_loss_uses_true_adj_not_pair_y(self):
        d = self.model.dim_h
        batch = SimpleNamespace()
        batch.x_raw = None

        h_out = torch.zeros(2, d)
        z_out = torch.randn(1, 2, 2, d)
        node_mask = torch.tensor([[True, True]])
        transport = torch.ones(2, 2, 2) / 4.0
        cost = torch.ones(2, 2, 2)
        atom_mask = torch.tensor([False, False])
        motif_block_mask = torch.tensor([False, False])
        motif_id = torch.tensor([0, 0], dtype=torch.long)
        true_adj_dense = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float)
        perturbed_pairs = {
            "b": torch.tensor([0], dtype=torch.long),
            "i": torch.tensor([0], dtype=torch.long),
            "j": torch.tensor([1], dtype=torch.long),
            "y": torch.tensor([0.0]),  # intentionally opposite to true_adj
        }

        losses = self.model._compute_pretrain_losses(
            batch=batch,
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
        )

        logit = self.model.edge_decoder(z_out).squeeze(-1)[0, 0, 1]
        expected_true_adj = torch.nn.BCEWithLogitsLoss()(
            logit.unsqueeze(0), torch.tensor([1.0])
        )
        expected_pair_y = torch.nn.BCEWithLogitsLoss()(
            logit.unsqueeze(0), torch.tensor([0.0])
        )
        torch.testing.assert_close(losses["edge_denoise"], expected_true_adj)
        self.assertFalse(torch.allclose(losses["edge_denoise"], expected_pair_y))

    def test_sinkhorn_eps_must_be_positive(self):
        cost = torch.ones(2, 3, 4)
        with self.assertRaises(ValueError):
            sinkhorn_transport(cost, eps=0.0, n_iters=3, log_domain=True)

    def test_epoch_weight_rolling_save_keeps_last_three(self):
        cfg.otformer.pretrain.save_epoch_weights = True
        cfg.otformer.pretrain.keep_last_epoch_weights = 3
        with tempfile.TemporaryDirectory() as td:
            cfg.run_dir = td
            for epoch in range(5):
                _save_epoch_weights_rolling(self.model, epoch)
            save_dir = Path(td) / "pretrain_weights"
            files = sorted(save_dir.glob("otformer_epoch_*.pt"))
            self.assertEqual(len(files), 3)
            self.assertEqual(
                [p.name for p in files],
                [
                    "otformer_epoch_0002.pt",
                    "otformer_epoch_0003.pt",
                    "otformer_epoch_0004.pt",
                ],
            )


if __name__ == "__main__":
    ut.main()
