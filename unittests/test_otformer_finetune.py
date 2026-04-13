import unittest as ut
from types import SimpleNamespace
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from yacs.config import CfgNode
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg

import gps  # noqa: F401, register modules
from gps.head.otformer_finetune_head import OTFormerFineTuneHead
from gps.loss.asymmetric_loss import (
    asymmetric_loss,
    bce_with_logits_finetune,
    mse_finetune,
)
from gps.finetuning import (
    get_final_pretrained_ckpt,
    compare_cfg,
    load_pretrained_model_cfg,
    init_model_from_pretrained,
)


class _Args:
    cfg_file = "configs/OTFormer/zinc-OTFormer-pretrain.yaml"
    opts = []
    repeat = 1
    mark_done = False


class TestOTFormerFineTuneHead(ut.TestCase):
    def setUp(self):
        set_cfg(cfg)
        load_cfg(cfg, _Args)
        cfg.dataset.node_encoder = False
        cfg.dataset.edge_encoder = False
        cfg.gnn.dim_inner = 16
        cfg.gnn.layers_pre_mp = 0
        cfg.otformer.motif.memory_size = 8
        cfg.otformer.finetune.readout_dim = 32
        cfg.otformer.finetune.cls_hidden = 16
        cfg.otformer.finetune.reg_hidden = 16
        cfg.otformer.dropout = 0.0

    def _make_batch(self, n_nodes, n_graphs, n_max):
        batch = SimpleNamespace()
        batch.x = torch.randn(n_nodes, cfg.gnn.dim_inner)
        batch.batch = torch.repeat_interleave(
            torch.arange(n_graphs), n_nodes // n_graphs
        )
        batch.y = torch.randn(n_graphs, 1)
        batch.otformer_aux = {
            "transport": torch.ones(n_nodes, 4, cfg.otformer.motif.memory_size)
            / (4 * cfg.otformer.motif.memory_size),
            "z_out": torch.randn(n_graphs, n_max, n_max, cfg.gnn.dim_inner),
            "node_mask": torch.ones(n_graphs, n_max, dtype=torch.bool),
        }
        return batch

    def test_classification_head_output_shape(self):
        cfg.dataset.task_type = "classification"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        batch = self._make_batch(n_nodes=20, n_graphs=4, n_max=5)
        pred, label = head(batch)
        self.assertEqual(pred.shape, (4, 1))

    def test_multilabel_head_output_shape(self):
        cfg.dataset.task_type = "classification_multilabel"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=12)
        batch = self._make_batch(n_nodes=20, n_graphs=4, n_max=5)
        pred, label = head(batch)
        self.assertEqual(pred.shape, (4, 12))

    def test_regression_head_output_shape(self):
        cfg.dataset.task_type = "regression"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        batch = self._make_batch(n_nodes=20, n_graphs=4, n_max=5)
        pred, label = head(batch)
        self.assertEqual(pred.shape, (4, 1))

    def test_cls_activation_sigmoid(self):
        cfg.dataset.task_type = "classification"
        cfg.otformer.finetune.cls_activation = "sigmoid"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        batch = self._make_batch(n_nodes=20, n_graphs=4, n_max=5)
        pred, _ = head(batch)
        self.assertTrue(
            (batch.graph_pred_prob >= 0).all() and (batch.graph_pred_prob <= 1).all()
        )

    def test_cls_activation_softmax(self):
        cfg.dataset.task_type = "classification"
        cfg.otformer.finetune.cls_activation = "softmax"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=5)
        batch = self._make_batch(n_nodes=20, n_graphs=4, n_max=5)
        pred, _ = head(batch)
        self.assertEqual(pred.shape, (4, 5))
        self.assertTrue((batch.graph_pred_prob >= 0).all())
        sums = batch.graph_pred_prob.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4), rtol=1e-5, atol=1e-5)

    def test_cls_activation_none(self):
        cfg.dataset.task_type = "classification"
        cfg.otformer.finetune.cls_activation = "none"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        self.assertIsInstance(head.output_activation, nn.Identity)
        batch = self._make_batch(n_nodes=20, n_graphs=4, n_max=5)
        pred, _ = head(batch)
        self.assertEqual(pred.shape, (4, 1))

    def test_cls_activation_invalid_raises(self):
        cfg.dataset.task_type = "classification"
        cfg.otformer.finetune.cls_activation = "invalid_act"
        with self.assertRaises(ValueError):
            OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)

    def test_mlp_activation_relu(self):
        cfg.dataset.task_type = "classification"
        cfg.otformer.finetune.cls_mlp_activation = "relu"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        mlp_act = head.task_head[1]
        self.assertIsInstance(mlp_act, nn.ReLU)

    def test_mlp_activation_leaky_relu(self):
        cfg.dataset.task_type = "classification"
        cfg.otformer.finetune.cls_mlp_activation = "leaky_relu"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        mlp_act = head.task_head[1]
        self.assertIsInstance(mlp_act, nn.LeakyReLU)

    def test_mlp_activation_swish(self):
        cfg.dataset.task_type = "classification"
        cfg.otformer.finetune.cls_mlp_activation = "swish"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        self.assertIsNotNone(head.task_head[1])

    def test_mlp_activation_gelu(self):
        cfg.dataset.task_type = "classification"
        cfg.otformer.finetune.cls_mlp_activation = "gelu"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        mlp_act = head.task_head[1]
        self.assertIsInstance(mlp_act, nn.GELU)

    def test_reg_mlp_activation_relu(self):
        cfg.dataset.task_type = "regression"
        cfg.otformer.finetune.reg_mlp_activation = "relu"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        mlp_act = head.task_head[1]
        self.assertIsInstance(mlp_act, nn.ReLU)

    def test_ot_readout_aggregation(self):
        cfg.dataset.task_type = "classification"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        n_nodes = 20
        n_graphs = 4
        K = cfg.otformer.motif.memory_size
        batch = self._make_batch(n_nodes=n_nodes, n_graphs=n_graphs, n_max=5)

        transport = torch.zeros(n_nodes, 4, K)
        transport[:, :, 0] = 1.0 / 4
        batch.otformer_aux["transport"] = transport

        r_ot = head._compute_ot_readout(transport, batch.batch)
        self.assertEqual(r_ot.shape, (n_graphs, cfg.otformer.motif.memory_size))

    def test_node_readout_aggregation(self):
        cfg.dataset.task_type = "classification"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        n_nodes = 20
        n_graphs = 4
        batch = self._make_batch(n_nodes=n_nodes, n_graphs=n_graphs, n_max=5)

        r_h = head._compute_node_readout(batch.x, batch.batch)
        self.assertEqual(r_h.shape, (n_graphs, 2 * cfg.gnn.dim_inner))

    def test_pair_readout_with_masking(self):
        cfg.dataset.task_type = "classification"
        head = OTFormerFineTuneHead(dim_in=cfg.gnn.dim_inner, dim_out=1)
        n_graphs = 2
        n_max = 4
        z_out = torch.ones(n_graphs, n_max, n_max, cfg.gnn.dim_inner)
        node_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]],
            dtype=torch.bool,
        )
        r_z = head._compute_pair_readout(z_out, node_mask)
        self.assertEqual(r_z.shape, (n_graphs, cfg.gnn.dim_inner))
        self.assertFalse(torch.isnan(r_z).any())


class TestASLLoss(ut.TestCase):
    def setUp(self):
        set_cfg(cfg)
        load_cfg(cfg, _Args)
        cfg.otformer.finetune.asl_gamma_pos = 1.0
        cfg.otformer.finetune.asl_gamma_neg = 4.0
        cfg.otformer.finetune.asl_clip = 0.05

    def test_asl_returns_none_for_wrong_loss_fun(self):
        cfg.model.loss_fun = "cross_entropy"
        pred = torch.tensor([0.5])
        true = torch.tensor([1.0])
        result = asymmetric_loss(pred, true)
        self.assertIsNone(result)

    def test_asl_loss_positive_sample(self):
        cfg.model.loss_fun = "asymmetric_loss"
        cfg.dataset.task_type = "classification_multilabel"
        pred = torch.tensor([[0.9]])
        true = torch.tensor([[1.0]])
        loss, _ = asymmetric_loss(pred, true)
        self.assertGreater(loss.item(), 0)
        self.assertLess(loss.item(), 10)

    def test_asl_loss_negative_sample_with_clipping(self):
        cfg.model.loss_fun = "asymmetric_loss"
        cfg.dataset.task_type = "classification_multilabel"
        pred = torch.tensor([[0.1]])
        true = torch.tensor([[0.0]])
        loss, _ = asymmetric_loss(pred, true)
        self.assertGreater(loss.item(), 0)

    def test_asl_handles_nan_labels(self):
        cfg.model.loss_fun = "asymmetric_loss"
        cfg.dataset.task_type = "classification_multilabel"
        pred = torch.tensor([[0.5, 0.7, 0.3]])
        true = torch.tensor([[1.0, float("nan"), 0.0]])
        loss, _ = asymmetric_loss(pred, true)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)

    def test_bce_finetune_multilabel(self):
        cfg.model.loss_fun = "bce_with_logits_finetune"
        cfg.dataset.task_type = "classification_multilabel"
        pred = torch.tensor([[0.5, 0.8]])
        true = torch.tensor([[1.0, 0.0]])
        loss, _ = bce_with_logits_finetune(pred, true)
        self.assertGreater(loss.item(), 0)

    def test_mse_finetune_regression(self):
        cfg.model.loss_fun = "mse_finetune"
        cfg.dataset.task_type = "regression"
        pred = torch.tensor([[1.5]])
        true = torch.tensor([[2.0]])
        loss, _ = mse_finetune(pred, true)
        torch.testing.assert_close(loss, torch.tensor(0.25))

    def test_mse_returns_none_for_non_regression(self):
        cfg.dataset.task_type = "classification"
        pred = torch.tensor([[1.5]])
        true = torch.tensor([[2.0]])
        result = mse_finetune(pred, true)
        self.assertIsNone(result)


class TestFinetuningUtils(ut.TestCase):
    def test_get_final_pretrained_ckpt(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td) / "0" / "ckpt"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "1.ckpt").touch()
            (ckpt_dir / "5.ckpt").touch()
            (ckpt_dir / "3.ckpt").touch()
            result = get_final_pretrained_ckpt(str(ckpt_dir))
            self.assertEqual(result, str(ckpt_dir / "5.ckpt"))

    def test_get_final_pretrained_ckpt_missing_dir(self):
        with self.assertRaises(FileNotFoundError):
            get_final_pretrained_ckpt("/nonexistent/path/ckpt")

    def test_get_final_pretrained_ckpt_supports_gps_snapshots(self):
        with tempfile.TemporaryDirectory() as td:
            pretrain_dir = Path(td) / "0" / "pretrain_weights"
            pretrain_dir.mkdir(parents=True)
            (pretrain_dir / "gps_epoch_0002.pt").touch()
            (pretrain_dir / "gps_epoch_0007.pt").touch()
            (pretrain_dir / "gps_epoch_0005.pt").touch()
            ckpt_dir = Path(td) / "0" / "ckpt"
            result = get_final_pretrained_ckpt(str(ckpt_dir))
            self.assertEqual(result, str(pretrain_dir / "gps_epoch_0007.pt"))

    def test_compare_cfg_strict_raises(self):
        set_cfg(cfg)
        cfg.otformer.finetune.readout_dim = 32
        other_cfg = CfgNode()
        other_cfg.otformer = CfgNode()
        other_cfg.otformer.finetune = CfgNode()
        other_cfg.otformer.finetune.readout_dim = 99
        with self.assertRaises(ValueError):
            compare_cfg(cfg, other_cfg, "otformer.finetune.readout_dim", strict=True)

    def test_compare_cfg_non_strict_warns(self):
        set_cfg(cfg)
        cfg.otformer.finetune.readout_dim = 64
        other_cfg = CfgNode()
        other_cfg.otformer = CfgNode()
        other_cfg.otformer.finetune = CfgNode()
        other_cfg.otformer.finetune.readout_dim = 128
        with self.assertLogs(level="WARNING"):
            compare_cfg(cfg, other_cfg, "otformer.finetune.readout_dim", strict=False)

    def test_load_pretrained_model_cfg_relaxes_encoder_check_for_gps_finetune(self):
        set_cfg(cfg)
        cfg.model.type = "GPSModel"
        cfg.gps.pretrain.enable = False
        cfg.gps.finetune.enable = True
        cfg.dataset.node_encoder = True
        cfg.dataset.node_encoder_name = "Atom+RWSE"
        cfg.dataset.node_encoder_bn = False
        cfg.dataset.edge_encoder = True
        cfg.dataset.edge_encoder_name = "Bond"
        cfg.dataset.edge_encoder_bn = False
        cfg.gnn.head = "san_graph"
        cfg.gnn.layers_post_mp = 3
        cfg.gnn.act = "relu"
        cfg.gnn.dropout = 0.0
        cfg.gt = CfgNode()
        cfg.gt.layer_type = "CustomGatedGCN+Transformer"

        with tempfile.TemporaryDirectory() as td:
            cfg.pretrained.dir = td
            config_path = Path(td) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  type: GPSModel",
                        "  graph_pooling: mean",
                        "  edge_decoding: dot",
                        "dataset:",
                        "  node_encoder: true",
                        "  node_encoder_name: TypeDictNode+RWSE",
                        "  node_encoder_bn: false",
                        "  edge_encoder: true",
                        "  edge_encoder_name: TypeDictEdge",
                        "  edge_encoder_bn: false",
                        "gt:",
                        "  layer_type: GINE+Transformer",
                        "gnn:",
                        "  head: san_graph",
                        "  layers_post_mp: 3",
                        "  act: relu",
                        "  dropout: 0.0",
                        "posenc_RWSE:",
                        "  enable: true",
                        "  dim_pe: 28",
                        "  model: Linear",
                    ]
                )
            )

            loaded_cfg = load_pretrained_model_cfg(cfg)
            self.assertEqual(loaded_cfg.gnn.head, "san_graph")
            self.assertEqual(loaded_cfg.gt.layer_type, "GINE+Transformer")


class TestInitModelFromPretrained(ut.TestCase):
    def setUp(self):
        set_cfg(cfg)
        load_cfg(cfg, _Args)
        cfg.dataset.node_encoder = False
        cfg.dataset.edge_encoder = False
        cfg.gnn.dim_inner = 16
        cfg.gnn.layers_pre_mp = 0
        cfg.gnn.head = "san_graph"
        cfg.otformer.pretrain.enable = False
        cfg.otformer.finetune.enable = False
        self.model = torch.nn.Module()
        self.model.body = torch.nn.Linear(16, 16)
        self.model.post_mp = torch.nn.Linear(16, 1)

    def _save_ckpt(self, path, include_head=True):
        state = {"body.weight": torch.randn(16, 16), "body.bias": torch.randn(16)}
        if include_head:
            state["post_mp.weight"] = torch.randn(1, 16)
            state["post_mp.bias"] = torch.randn(1)
        torch.save({"model_state": state}, path)

    def test_load_with_reset_head(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "model.ckpt"
            self._save_ckpt(ckpt_path, include_head=True)
            model = torch.nn.Module()
            model.body = torch.nn.Linear(16, 16)
            model.post_mp = torch.nn.Linear(16, 1)
            result = init_model_from_pretrained(
                model,
                "",
                freeze_main=False,
                reset_prediction_head=True,
                seed=0,
                weights_path=str(ckpt_path),
            )
            self.assertTrue(result.post_mp.weight.requires_grad)

    def test_freeze_main(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "model.ckpt"
            self._save_ckpt(ckpt_path, include_head=True)
            model = torch.nn.Module()
            model.body = torch.nn.Linear(16, 16)
            model.post_mp = torch.nn.Linear(16, 1)
            result = init_model_from_pretrained(
                model,
                "",
                freeze_main=True,
                reset_prediction_head=True,
                seed=0,
                weights_path=str(ckpt_path),
            )
            self.assertFalse(result.body.weight.requires_grad)
            self.assertTrue(result.post_mp.weight.requires_grad)

    def test_missing_weights_path_raises(self):
        with tempfile.TemporaryDirectory() as td:
            model = torch.nn.Module()
            model.body = torch.nn.Linear(16, 16)
            model.post_mp = torch.nn.Linear(16, 1)
            with self.assertRaises(FileNotFoundError):
                init_model_from_pretrained(
                    model,
                    "",
                    weights_path=f"{td}/nonexistent.ckpt",
                )


if __name__ == "__main__":
    ut.main()
