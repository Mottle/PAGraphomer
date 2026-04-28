import unittest as ut

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg

import gps  # noqa: F401
from gps.encoder.rrwp_encoder import RRWPPairEncoder
from gps.layer.scaled_range_former_layer import (
    ScaledRangeFormerAttention,
    ScaledRangeFormerBypassLayer,
    ScaledRangeFormerMixedLayer,
    ScaledRangeFormerSequentialLayer,
    ScaledRangeFormerSharedLayer,
)


class _Args:
    cfg_file = "configs/GRIT/zinc-ScaledRangeFormerMixed.yaml"
    opts = []
    repeat = 1
    mark_done = False


def _toy_batch(dim_h=32, rrwp_dim=11):
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 0, 0, 1, 2, 2],
            [1, 0, 2, 1, 0, 2, 1, 0, 2],
        ],
        dtype=torch.long,
    )
    rrwp_val = torch.rand(edge_index.size(1), rrwp_dim)
    data = Data(
        x=torch.randn(3, dim_h),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.size(1), dim_h),
        pair_index=edge_index,
        pair_attr=torch.randn(edge_index.size(1), dim_h),
        rrwp=torch.randn(3, rrwp_dim),
        rrwp_index=edge_index,
        rrwp_val=rrwp_val,
        srf_spd_index=edge_index,
        srf_spd_val=torch.tensor([1, 1, 1, 1, 0, 2, 0, 2, 0], dtype=torch.long),
        deg=torch.tensor([1, 2, 1], dtype=torch.long),
        log_deg=torch.log(torch.tensor([2.0, 3.0, 2.0])),
        y=torch.tensor([0.0]),
    )
    return Batch.from_data_list([data])


class TestScaledRangeFormerLayer(ut.TestCase):
    def setUp(self):
        set_cfg(cfg)
        load_cfg(cfg, _Args)

    def _assert_layer_forward(self, layer_cls):
        batch = _toy_batch(dim_h=cfg.gt.dim_hidden, rrwp_dim=cfg.posenc_RRWP.walk_length)
        layer = layer_cls(
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
        self.assertEqual(out.pair_attr.shape[1], cfg.gt.dim_hidden)
        self.assertEqual(out.pair_index.shape[0], 2)
        self.assertEqual(out.pair_attr.shape[0], out.pair_index.shape[1])

    def test_rrwp_pair_encoder_injects_edge_and_flag(self):
        batch = _toy_batch(dim_h=cfg.gt.dim_hidden, rrwp_dim=cfg.posenc_RRWP.walk_length)
        encoder = RRWPPairEncoder(
            emb_dim=cfg.posenc_RRWP.walk_length,
            out_dim=cfg.gt.dim_hidden,
            edge_dim=cfg.gt.dim_hidden,
            rrwp_name="rrwp",
            pad_to_full_graph=True,
            inject_edge_attr=True,
            add_adj_indicator=True,
        )
        out = encoder(batch)
        self.assertTrue(hasattr(out, "pair_index"))
        self.assertTrue(hasattr(out, "pair_attr"))
        self.assertEqual(out.pair_index.shape[0], 2)
        self.assertEqual(out.pair_attr.shape[1], cfg.gt.dim_hidden)
        self.assertGreaterEqual(out.pair_index.shape[1], out.edge_index.shape[1])

    def test_distance_masks_are_built_from_spd_thresholds(self):
        batch = _toy_batch(dim_h=cfg.gt.dim_hidden, rrwp_dim=cfg.posenc_RRWP.walk_length)
        attn = ScaledRangeFormerAttention(
            dim_h=cfg.gt.dim_hidden,
            num_heads=cfg.gt.n_heads,
            attn_dropout=cfg.gt.attn_dropout,
            formulation="B",
            cfg=cfg.gt,
        )
        pair_index, pair_attr = attn._coalesce_pair_graph(batch)
        spd_full = attn._align_spd(batch, pair_index, pair_attr)
        scale_values = attn._compute_scale_values(batch, spd_full, pair_index)
        self.assertEqual(len(scale_values), len(cfg.gt.msrrwp.thresholds))
        self.assertTrue(scale_values[-1].all())

    def test_soft_mask_uses_score_times_binary_mask(self):
        batch = _toy_batch(dim_h=cfg.gt.dim_hidden, rrwp_dim=cfg.posenc_RRWP.walk_length)
        cfg.gt.msrrwp.mask_mode = "soft"
        attn = ScaledRangeFormerAttention(
            dim_h=cfg.gt.dim_hidden,
            num_heads=cfg.gt.n_heads,
            attn_dropout=cfg.gt.attn_dropout,
            formulation="B",
            cfg=cfg.gt,
        )
        score = torch.ones(batch.pair_index.size(1), cfg.gt.n_heads, 1)
        mask = torch.tensor([1, 0, 1, 0, 1, 1, 1, 1, 1], dtype=torch.float)
        masked_score, edge_coeff = attn._apply_structure(score, mask)
        self.assertEqual(float(masked_score[1, 0, 0]), 0.0)
        self.assertEqual(float(masked_score[0, 0, 0]), 1.0)
        self.assertTrue(torch.equal(edge_coeff, mask))

    def test_percentiles_are_computed_per_graph(self):
        data1 = _toy_batch(dim_h=cfg.gt.dim_hidden, rrwp_dim=cfg.posenc_RRWP.walk_length)
        data2 = _toy_batch(dim_h=cfg.gt.dim_hidden, rrwp_dim=cfg.posenc_RRWP.walk_length)
        data_list = Batch.to_data_list(data1) + Batch.to_data_list(data2)
        data_list[1].srf_spd_val = torch.tensor([5, 5, 5, 5, 0, 7, 0, 7, 0], dtype=torch.long)
        batch = Batch.from_data_list(data_list)
        attn = ScaledRangeFormerAttention(
            dim_h=cfg.gt.dim_hidden,
            num_heads=cfg.gt.n_heads,
            attn_dropout=cfg.gt.attn_dropout,
            formulation="B",
            cfg=cfg.gt,
        )
        pair_index, pair_attr = attn._coalesce_pair_graph(batch)
        spd_full = attn._align_spd(batch, pair_index, pair_attr)
        scale_values = attn._compute_scale_values(batch, spd_full, pair_index)
        first_graph_pairs = batch.batch[pair_index[0]] == 0
        second_graph_pairs = batch.batch[pair_index[0]] == 1
        self.assertFalse(torch.equal(scale_values[0][first_graph_pairs], scale_values[0][second_graph_pairs]))

    def test_b_plus_uses_per_scale_alphas(self):
        self.assertEqual(list(cfg.gt.msrrwp.alphas), [1.0, 1.5, 2.0])
        attn = ScaledRangeFormerAttention(
            dim_h=cfg.gt.dim_hidden,
            num_heads=cfg.gt.n_heads,
            attn_dropout=cfg.gt.attn_dropout,
            formulation="B",
            cfg=cfg.gt,
        )
        self.assertEqual(attn.alphas, [1.0, 1.5, 2.0])

    def test_formulation_a_forward(self):
        self._assert_layer_forward(ScaledRangeFormerSharedLayer)

    def test_formulation_b_forward(self):
        self._assert_layer_forward(ScaledRangeFormerMixedLayer)

    def test_formulation_c_forward(self):
        self._assert_layer_forward(ScaledRangeFormerSequentialLayer)

    def test_formulation_d_forward(self):
        self._assert_layer_forward(ScaledRangeFormerBypassLayer)


if __name__ == "__main__":
    ut.main()
