import math
import unittest

import torch
import torch.nn as nn

from gps.layer.srf_rum_mani_layer import MotifExtractor, SRFxRUM_MANI_Layer


class DummySRFAttention(nn.Module):
    """Minimal stand-in for ScaledRangeFormerAttention in smoke tests."""

    def __init__(self, dim_h, num_heads):
        super().__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads

    def forward(self, batch):
        # Simply return identity-like outputs
        h = batch.x
        num_pairs = batch.pair_attr.size(0)
        h_out = torch.zeros(h.size(0), self.dim_h, device=h.device, dtype=h.dtype)
        e_out = torch.zeros(num_pairs, self.dim_h, device=h.device, dtype=h.dtype)
        return h_out, e_out


class TestMotifExtractor(unittest.TestCase):
    def test_shape_and_softmax(self):
        N, D, K = 8, 16, 4
        extractor = MotifExtractor(D, K, temperature=1.0)
        h_rum = torch.randn(N, D)
        score, emb = extractor(h_rum)

        self.assertEqual(score.shape, (N, K))
        self.assertEqual(emb.shape, (N, D))

        # Softmax over prototypes
        self.assertTrue(torch.allclose(score.sum(dim=-1), torch.ones(N), atol=1e-5))
        self.assertTrue((score >= 0).all() and (score <= 1).all())


class TestSRFxRUM_MANI_LayerSmoke(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.N = 6
        self.D = 16
        self.num_heads = 4
        self.K = 4

    def _make_batch(self):
        """Construct a minimal Batch-like object with required attributes."""
        from torch_geometric.data import Batch, Data

        # Two small graphs: 3 nodes each
        g1 = Data(
            x=torch.randn(3, self.D),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
            edge_attr=torch.randn(4, self.D),
        )
        g2 = Data(
            x=torch.randn(3, self.D),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
            edge_attr=torch.randn(4, self.D),
        )
        batch = Batch.from_data_list([g1, g2])

        # Pair graph (simplified: same as edge_index for this smoke test)
        batch.pair_index = batch.edge_index.clone()
        batch.pair_attr = batch.edge_attr.clone()

        return batch

    def _make_layer(self, use_dummy_srf=True):
        srf_cfg = type(
            "Cfg",
            (),
            {
                "formulation": "B",
                "attn_dropout": 0.0,
                "msrrwp": type(
                    "MsCfg",
                    (),
                    {
                        "rrwp_name": "rrwp",
                        "use_spd": False,
                        "spd_name": "srf_spd",
                        "scale_mode": "percentile",
                        "mask_mode": "hard",
                        "soft_eps": 1e-6,
                        "hard_eps": 1e-9,
                        "weight_in_softmax": False,
                        "learnable_weights": True,
                        "inner_residual": True,
                        "inner_norm": True,
                        "thresholds": [1.0],
                        "alphas": [1.0],
                    },
                )(),
                "attn": type(
                    "AttnCfg",
                    (),
                    {
                        "use_bias": False,
                        "clamp": 5.0,
                        "act": "relu",
                    },
                )(),
            },
        )()

        rum_cfg = {
            "depth": 1,
            "num_samples": 2,
            "length": 2,
            "dropout": 0.0,
            "use_edge_features": False,
            "self_supervise": False,
            "binary": False,
        }

        layer = SRFxRUM_MANI_Layer(
            dim=self.D,
            num_heads=self.num_heads,
            srf_cfg=srf_cfg,
            rum_cfg=rum_cfg,
            num_prototypes=self.K,
            motif_temperature=1.0,
            alpha=1.0,
            dropout=0.0,
            act="relu",
            layer_norm=False,
            batch_norm=False,
            residual=True,
        )

        if use_dummy_srf:
            layer.srf_attn = DummySRFAttention(self.D, self.num_heads)

        return layer.to(self.device)

    def test_forward_shapes(self):
        batch = self._make_batch()
        layer = self._make_layer(use_dummy_srf=True)

        out_batch, motif_score = layer(batch)

        self.assertEqual(out_batch.x.shape, (self.N, self.D))
        self.assertEqual(motif_score.shape, (self.N, self.K))

    def test_motif_score_properties(self):
        batch = self._make_batch()
        layer = self._make_layer(use_dummy_srf=True)
        _, motif_score = layer(batch)

        # Softmax properties
        self.assertTrue(
            torch.allclose(motif_score.sum(dim=-1), torch.ones(self.N), atol=1e-5)
        )
        self.assertTrue((motif_score >= 0).all() and (motif_score <= 1).all())

    def test_gate_range(self):
        batch = self._make_batch()
        layer = self._make_layer(use_dummy_srf=True)
        out_batch, _ = layer(batch)

        # Gate outputs sigmoid, so values in (0, 1)
        # We don't directly expose gate, but fused output should be in feature space
        self.assertFalse(torch.isnan(out_batch.x).any())
        self.assertFalse(torch.isinf(out_batch.x).any())


if __name__ == "__main__":
    unittest.main()
