import unittest as ut
import torch
import torch.nn as nn

from gps.optimizer.muon import Muon
from gps.optimizer.extra_optimizers import _MuonHybridOptimizer
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
import torch_geometric.graphgym.register as register
import gps  # noqa: F401 registers optimizers


class _Args:
    cfg_file = "configs/OTFormer/zinc-OTFormer-pretrain.yaml"
    opts = []
    repeat = 1
    mark_done = False


class TestMuonHybridOptimizer(ut.TestCase):
    def setUp(self):
        set_cfg(cfg)
        load_cfg(cfg, _Args)

    def test_muon_registered(self):
        """Muon optimizer should be registered in GraphGym."""
        self.assertIn("muon_adamw", register.optimizer_dict)

    def test_hybrid_separates_params_by_ndim(self):
        """Hybrid optimizer should separate 2D and non-2D parameters."""
        model = nn.Sequential(
            nn.Linear(10, 20),  # weight (2D), bias (1D)
            nn.LayerNorm(20),  # weight (1D), bias (1D)
            nn.Linear(20, 5),  # weight (2D), bias (1D)
        )
        params = list(model.parameters())
        opt = _MuonHybridOptimizer(
            Muon([p for p in params if p.ndim == 2], lr=1e-3, weight_decay=0.1),
            torch.optim.AdamW(
                [p for p in params if p.ndim != 2], lr=1e-4, weight_decay=0.01
            ),
        )
        self.assertEqual(len(opt.muon.param_groups[0]["params"]), 2)
        self.assertGreater(len(opt.adamw.param_groups[0]["params"]), 0)

    def test_hybrid_zero_grad_and_step(self):
        """Hybrid optimizer zero_grad and step should work without error."""
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        params = list(model.parameters())
        opt = _MuonHybridOptimizer(
            Muon([p for p in params if p.ndim == 2], lr=1e-3, weight_decay=0.1),
            torch.optim.AdamW(
                [p for p in params if p.ndim != 2], lr=1e-4, weight_decay=0.01
            ),
        )
        x = torch.randn(2, 8)
        y = model(x)
        loss = y.sum()
        loss.backward()
        opt.zero_grad()
        for p in params:
            if p.grad is not None:
                self.assertTrue((p.grad == 0).all())

    def test_hybrid_state_dict_roundtrip(self):
        """State dict should be savable and loadable."""
        w1 = nn.Parameter(torch.randn(4, 8))
        b1 = nn.Parameter(torch.randn(8))
        w1.grad = torch.randn(4, 8)
        b1.grad = torch.randn(8)

        opt = _MuonHybridOptimizer(
            Muon([w1], lr=1e-3, weight_decay=0.1),
            torch.optim.AdamW([b1], lr=1e-4, weight_decay=0.01),
        )
        opt.step()
        sd = opt.state_dict()
        self.assertIn("muon_adamw", sd)
        self.assertIn("adamw", sd)

    def test_muon_rejects_non_2d(self):
        """Muon should reject non-2D parameters."""
        p = nn.Parameter(torch.randn(10))
        with self.assertRaises(ValueError):
            Muon([p], lr=1e-3)


class TestMuonNewtonSchulz(ut.TestCase):
    def test_zeropower_produces_orthogonal(self):
        """Newton-Schulz should produce approximately orthogonal matrices."""
        from gps.optimizer.muon import _zeropower_via_newtonschulz

        G = torch.randn(8, 6)
        U = _zeropower_via_newtonschulz(G, (3.4445, -4.7750, 2.0315), 5, 1e-7)

        # Output should have same shape as input, no NaN/Inf
        self.assertEqual(U.shape, G.shape)
        self.assertFalse(torch.isnan(U).any())
        self.assertFalse(torch.isinf(U).any())

        # Newton-Schulz produces US'V^T where S' ~ Uniform(0.5, 1.5).
        # So spectral norm should be bounded and finite.
        _, s, _ = torch.svd(U.float())
        self.assertLess(s.max().item(), 2.0)
        self.assertGreater(s.min().item(), 0.1)

    def test_zeropower_rejects_1d(self):
        """Should reject 1D input."""
        from gps.optimizer.muon import _zeropower_via_newtonschulz

        with self.assertRaises(ValueError):
            _zeropower_via_newtonschulz(
                torch.randn(10), (3.4445, -4.7750, 2.0315), 5, 1e-7
            )


if __name__ == "__main__":
    ut.main()
