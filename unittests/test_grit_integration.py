import unittest as ut

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg

import gps  # noqa: F401
from gps.encoder.rrwp_encoder import RRWPLinearEdgeEncoder, RRWPLinearNodeEncoder
from gps.layer.grit_layer import GritTransformerLayer
from gps.network.grit_model import GritTransformer


class _Args:
    cfg_file = "configs/GRIT/zinc-GRIT-RRWP.yaml"
    opts = []
    repeat = 1
    mark_done = False


def _toy_batch():
    data = Data(
        x=torch.tensor([[1], [2], [3]], dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.tensor([[1], [1], [2], [2]], dtype=torch.long),
        y=torch.tensor([0.0]),
        rrwp=torch.randn(3, 21),
        rrwp_index=torch.tensor(
            [[0, 1, 1, 2, 0, 1, 2], [0, 0, 1, 1, 2, 2, 2]], dtype=torch.long
        ),
        rrwp_val=torch.randn(7, 21),
        deg=torch.tensor([1, 2, 1], dtype=torch.long),
        log_deg=torch.log(torch.tensor([2.0, 3.0, 2.0])),
    )
    return Batch.from_data_list([data])


class TestGRITIntegration(ut.TestCase):
    def setUp(self):
        set_cfg(cfg)
        load_cfg(cfg, _Args)

    def test_rrwp_node_encoder_preserves_shape(self):
        batch = _toy_batch()
        batch.x = torch.randn(batch.num_nodes, cfg.gnn.dim_inner)
        enc = RRWPLinearNodeEncoder(21, cfg.gnn.dim_inner)
        out = enc(batch)
        self.assertEqual(out.x.shape, (batch.num_nodes, cfg.gnn.dim_inner))

    def test_rrwp_edge_encoder_pads_to_full_graph(self):
        batch = _toy_batch()
        batch.edge_attr = torch.randn(batch.edge_index.size(1), cfg.gnn.dim_inner)
        enc = RRWPLinearEdgeEncoder(21, cfg.gnn.dim_inner, pad_to_full_graph=True)
        out = enc(batch)
        self.assertEqual(out.edge_attr.shape[1], cfg.gnn.dim_inner)
        self.assertGreaterEqual(out.edge_index.size(1), batch.edge_index.size(1))

    def test_grit_layer_forward(self):
        batch = _toy_batch()
        batch.x = torch.randn(batch.num_nodes, cfg.gnn.dim_inner)
        batch.edge_attr = torch.randn(batch.edge_index.size(1), cfg.gnn.dim_inner)
        layer = GritTransformerLayer(
            in_dim=cfg.gt.dim_hidden,
            out_dim=cfg.gt.dim_hidden,
            num_heads=cfg.gt.n_heads,
            dropout=cfg.gt.dropout,
            attn_dropout=cfg.gt.attn_dropout,
            layer_norm=cfg.gt.layer_norm,
            batch_norm=cfg.gt.batch_norm,
            residual=True,
            act=cfg.gnn.act,
            norm_e=cfg.gt.attn.norm_e,
            O_e=cfg.gt.attn.O_e,
            cfg=cfg.gt,
        )
        out = layer(batch)
        self.assertEqual(out.x.shape, (batch.num_nodes, cfg.gt.dim_hidden))

    def test_grit_model_forward(self):
        model = GritTransformer(dim_in=1, dim_out=1)
        batch = _toy_batch()
        batch.edge_attr = batch.edge_attr.squeeze(-1)
        pred, true = model(batch)
        self.assertEqual(pred.shape[0], 1)
        self.assertEqual(true.shape[0], 1)

    def test_grit_model_pretrain_forward(self):
        cfg.grit.pretrain.enable = True
        model = GritTransformer(dim_in=1, dim_out=1)
        batch = _toy_batch()
        batch.edge_attr = batch.edge_attr.squeeze(-1)
        pred, true = model(batch)
        self.assertTrue(hasattr(batch, "gps_aux"))
        self.assertIn("loss", batch.gps_aux)
        self.assertIn("atom_logits", batch.gps_aux)
        self.assertIn("atom_mask", batch.gps_aux)
        self.assertEqual(pred.shape, true.shape)

    def test_grit_dual_view_depth_override_changes_backbone(self):
        model = GritTransformer(dim_in=1, dim_out=1)
        batch = _toy_batch()
        batch.edge_attr = batch.edge_attr.squeeze(-1)
        out_full = model._forward_backbone_with_dropout(
            batch.clone(), dropout_override=0.0, depth_override=len(model.layers)
        )
        out_shallow = model._forward_backbone_with_dropout(
            batch.clone(),
            dropout_override=0.0,
            depth_override=max(1, len(model.layers) - 1),
        )
        self.assertEqual(out_full.x.shape, out_shallow.x.shape)
        self.assertFalse(torch.allclose(out_full.x, out_shallow.x))


if __name__ == "__main__":
    ut.main()
