import tempfile
import unittest as ut
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg

import gps  # noqa: F401, register modules
from gps.network.gps_model import GPSModel
from gps.train.gps_pretrain import _save_epoch_weights_rolling


class _Args:
    cfg_file = "configs/GPS/zinc-GPS-pretrain.yaml"
    opts = []
    repeat = 1
    mark_done = False


class TestGPSPretrain(ut.TestCase):
    def setUp(self):
        set_cfg(cfg)
        load_cfg(cfg, _Args)
        cfg.dataset.node_encoder_name = "TypeDictNode"
        cfg.dataset.edge_encoder = True
        cfg.dataset.edge_encoder_name = "TypeDictEdge"
        cfg.posenc_RWSE.enable = False
        cfg.gnn.layers_pre_mp = 0
        cfg.gt.layers = 1
        cfg.gt.n_heads = 2
        cfg.gt.dim_hidden = 16
        cfg.gnn.dim_inner = 16
        cfg.gt.dropout = 0.0
        cfg.gt.attn_dropout = 0.0
        cfg.gps.pretrain.enable = True
        cfg.gps.pretrain.mask_ratio = 0.5
        self.model = GPSModel(dim_in=1, dim_out=1)

    def _make_batch(self):
        batch = SimpleNamespace()
        batch.x = torch.tensor([[1], [2], [3], [4]], dtype=torch.long)
        batch.edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        batch.edge_attr = torch.zeros(batch.edge_index.size(1), dtype=torch.long)
        batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long)
        batch.y = torch.tensor([[0.0]])
        return batch

    def test_pretrain_forward_populates_aux_loss(self):
        torch.manual_seed(0)
        batch = self._make_batch()
        pred, true = self.model(batch)

        self.assertEqual(pred.shape, true.shape)
        self.assertTrue(hasattr(batch, "gps_aux"))
        self.assertIn("loss", batch.gps_aux)
        self.assertIn("losses", batch.gps_aux)
        self.assertIn("atom_mask", batch.gps_aux)
        self.assertIn("atom_logits", batch.gps_aux)
        self.assertIn("edge_mask", batch.gps_aux)
        self.assertIn("edge_logits", batch.gps_aux)
        self.assertIn("edge_supervision_idx", batch.gps_aux)
        self.assertGreater(int(batch.gps_aux["atom_mask"].sum().item()), 0)
        self.assertGreater(int(batch.gps_aux["edge_mask"].sum().item()), 0)
        self.assertEqual(batch.gps_aux["atom_logits"].shape[0], batch.x.shape[0])
        self.assertEqual(batch.gps_aux["atom_logits"].shape[1], 28)
        self.assertEqual(
            batch.gps_aux["edge_logits"].shape[0], batch.edge_index.shape[1]
        )
        self.assertEqual(batch.gps_aux["edge_logits"].shape[1], 4)
        self.assertGreaterEqual(batch.gps_aux["edge_supervision_idx"].numel(), 1)
        self.assertLessEqual(batch.gps_aux["edge_supervision_idx"].numel(), 3)
        self.assertGreaterEqual(float(batch.gps_aux["loss"].item()), 0.0)

    def test_edge_mask_is_symmetric_for_bidirectional_edges(self):
        batch = self._make_batch()
        edge_mask, edge_supervision_idx = self.model._sample_edge_pretrain_mask(
            batch.edge_index,
            ratio=1.0,
            device=torch.device("cpu"),
        )
        self.assertTrue(edge_mask.all().item())
        self.assertEqual(edge_supervision_idx.numel(), 3)

        cfg.gps.pretrain.edge_mask_ratio = 0.0
        edge_mask, edge_supervision_idx = self.model._sample_edge_pretrain_mask(
            batch.edge_index,
            ratio=0.0,
            device=torch.device("cpu"),
        )
        self.assertFalse(edge_mask.any().item())
        self.assertEqual(edge_supervision_idx.numel(), 0)

    def test_zero_mask_ratio_returns_zero_mask(self):
        cfg.gps.pretrain.mask_ratio = 0.0
        mask = self.model._sample_pretrain_mask(4, torch.device("cpu"))
        self.assertFalse(mask.any().item())

    def test_epoch_weight_rolling_save_keeps_last_three(self):
        cfg.gps.pretrain.save_epoch_weights = True
        cfg.gps.pretrain.keep_last_epoch_weights = 3
        with tempfile.TemporaryDirectory() as td:
            cfg.run_dir = td
            for epoch in range(5):
                _save_epoch_weights_rolling(self.model, epoch)
            save_dir = Path(td) / "pretrain_weights"
            files = sorted(save_dir.glob("gps_epoch_*.pt"))
            self.assertEqual(len(files), 3)
            self.assertEqual(
                [p.name for p in files],
                [
                    "gps_epoch_0002.pt",
                    "gps_epoch_0003.pt",
                    "gps_epoch_0004.pt",
                ],
            )


if __name__ == "__main__":
    ut.main()
