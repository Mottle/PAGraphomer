import logging
import os
import os.path as osp
from glob import glob
import yaml

import torch
from torch_geometric.graphgym.config import set_cfg
from yacs.config import CfgNode


def _find_latest_pretrained_dir_filtered(base_dir="results", cfg_filter=None):
    """Find latest OTFormer pretraining dir with an optional config filter."""
    if not osp.exists(base_dir):
        return None

    candidates = []
    for exp_name in os.listdir(base_dir):
        exp_path = osp.join(base_dir, exp_name)
        cfg_file = osp.join(exp_path, "config.yaml")

        if osp.isdir(exp_path) and osp.isfile(cfg_file):
            try:
                with open(cfg_file, "r") as f:
                    exp_cfg = yaml.safe_load(f)

                model_type = exp_cfg.get("model", {}).get("type")
                is_pretrain = (
                    exp_cfg.get("otformer", {}).get("pretrain", {}).get("enable", False)
                )

                if model_type == "OTFormerModel" and is_pretrain:
                    if cfg_filter is not None and not cfg_filter(exp_cfg):
                        continue
                    # Check if any seed dir has either GraphGym ckpt (*.ckpt)
                    # or OTFormer rolling pretrain weights (*.pt).
                    has_weights = False
                    for item in os.listdir(exp_path):
                        seed_dir = osp.join(exp_path, item)
                        if not osp.isdir(seed_dir):
                            continue
                        ckpt_dir = osp.join(seed_dir, "ckpt")
                        pretrain_dir = osp.join(seed_dir, "pretrain_weights")
                        has_graphgym_ckpt = (
                            osp.isdir(ckpt_dir)
                            and len(glob(osp.join(ckpt_dir, "*.ckpt"))) > 0
                        )
                        has_pretrain_pt = (
                            osp.isdir(pretrain_dir)
                            and len(glob(osp.join(pretrain_dir, "otformer_epoch_*.pt")))
                            > 0
                        )
                        if has_graphgym_ckpt or has_pretrain_pt:
                            has_weights = True
                            break

                    if has_weights:
                        mtime = osp.getmtime(exp_path)
                        candidates.append((mtime, exp_path))
            except Exception:
                continue

    if not candidates:
        return None

    # Return the most recently modified directory
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_latest_pretrained_dir(base_dir="results"):
    """
    Search for the latest OTFormer pretrained model directory.
    Scans `base_dir` for experiments with `model.type: OTFormerModel`
    and `otformer.pretrain.enable: True`.
    """
    return _find_latest_pretrained_dir_filtered(base_dir=base_dir, cfg_filter=None)


def find_latest_zinc_pretrained_dir(base_dir="results"):
    """Search for the latest OTFormer pretraining run on ZINC."""

    def _is_zinc(exp_cfg):
        dataset = exp_cfg.get("dataset", {})
        dataset_name = str(dataset.get("name", "")).lower()
        dataset_format = str(dataset.get("format", "")).lower()
        # Support both:
        # - dataset.name: zinc
        # - PyG-ZINC configs where dataset.name is often "subset"
        return dataset_name == "zinc" or dataset_format == "pyg-zinc"

    return _find_latest_pretrained_dir_filtered(base_dir=base_dir, cfg_filter=_is_zinc)


def get_final_pretrained_ckpt(ckpt_dir):
    if osp.isdir(ckpt_dir):
        ckpt_files = glob(osp.join(ckpt_dir, "*.ckpt"))
        if ckpt_files:

            def _ckpt_epoch(path):
                return int(osp.basename(path).split(".")[0])

            return max(ckpt_files, key=_ckpt_epoch)

    # Fallback: OTFormer pretraining rolling snapshots.
    # Expected layout: {run_dir}/{seed}/pretrain_weights/otformer_epoch_XXXX.pt
    seed_dir = osp.dirname(ckpt_dir.rstrip("/"))
    pretrain_dir = osp.join(seed_dir, "pretrain_weights")
    pretrain_files = glob(osp.join(pretrain_dir, "otformer_epoch_*.pt"))
    if pretrain_files:

        def _pt_epoch(path):
            stem = osp.splitext(osp.basename(path))[0]
            # otformer_epoch_0007 -> 7
            return int(stem.split("_")[-1])

        return max(pretrain_files, key=_pt_epoch)

    raise FileNotFoundError(
        f"No pretrained weights found under '{ckpt_dir}' or '{pretrain_dir}'."
    )


def compare_cfg(cfg_main, cfg_secondary, field_name, strict=False):
    main_val, secondary_val = cfg_main, cfg_secondary
    for f in field_name.split("."):
        main_val = main_val[f]
        secondary_val = secondary_val[f]
    if main_val != secondary_val:
        if strict:
            raise ValueError(
                f"Main and pretrained configs must match on '{field_name}'"
            )
        else:
            logging.warning(
                f"Pretrained models '{field_name}' differs, using: {main_val}"
            )


def set_new_cfg_allowed(config, is_new_allowed):
    """Set YACS config (and recursively its subconfigs) to allow merging
    new keys from other configs.
    """
    config.__dict__[CfgNode.NEW_ALLOWED] = is_new_allowed
    for v in config.__dict__.values():
        if isinstance(v, CfgNode):
            set_new_cfg_allowed(v, is_new_allowed)
    for v in config.values():
        if isinstance(v, CfgNode):
            set_new_cfg_allowed(v, is_new_allowed)


def load_pretrained_model_cfg(cfg):
    if not cfg.pretrained.dir:
        raise ValueError(
            "pretrained.dir must be set when loading pretrained model config. "
            "Use pretrained.weights_path instead if you only want to load weights."
        )

    pretrained_cfg_fname = osp.join(cfg.pretrained.dir, "config.yaml")
    if not os.path.isfile(pretrained_cfg_fname):
        raise FileNotFoundError(
            f"Pretrained model config not found: {pretrained_cfg_fname}"
        )

    logging.info(f"[*] Updating cfg from pretrained model: {pretrained_cfg_fname}")

    pretrained_cfg = CfgNode()
    set_cfg(pretrained_cfg)
    set_new_cfg_allowed(pretrained_cfg, True)
    pretrained_cfg.merge_from_file(pretrained_cfg_fname)

    assert cfg.model.type in [
        "GPSModel",
        "Graphormer",
        "OTFormerModel",
    ], f"Fine-tuning regime is untested for model type: {cfg.model.type}"

    compare_cfg(cfg, pretrained_cfg, "model.type", strict=True)
    compare_cfg(cfg, pretrained_cfg, "model.graph_pooling")
    compare_cfg(cfg, pretrained_cfg, "model.edge_decoding")
    relax_dataset_encoder_check = cfg.model.type == "OTFormerModel" and bool(
        getattr(cfg.otformer.finetune, "enable", False)
    )
    compare_cfg(
        cfg,
        pretrained_cfg,
        "dataset.node_encoder",
        strict=not relax_dataset_encoder_check,
    )
    compare_cfg(
        cfg,
        pretrained_cfg,
        "dataset.node_encoder_name",
        strict=not relax_dataset_encoder_check,
    )
    compare_cfg(
        cfg,
        pretrained_cfg,
        "dataset.node_encoder_bn",
        strict=not relax_dataset_encoder_check,
    )
    compare_cfg(
        cfg,
        pretrained_cfg,
        "dataset.edge_encoder",
        strict=not relax_dataset_encoder_check,
    )
    compare_cfg(
        cfg,
        pretrained_cfg,
        "dataset.edge_encoder_name",
        strict=not relax_dataset_encoder_check,
    )
    compare_cfg(
        cfg,
        pretrained_cfg,
        "dataset.edge_encoder_bn",
        strict=not relax_dataset_encoder_check,
    )

    if cfg.model.type == "OTFormerModel":
        compare_cfg(cfg, pretrained_cfg, "gnn.dim_inner", strict=True)
        compare_cfg(cfg, pretrained_cfg, "otformer.motif.memory_size", strict=True)
        compare_cfg(cfg, pretrained_cfg, "otformer.num_heads", strict=True)
        compare_cfg(cfg, pretrained_cfg, "otformer.layers", strict=False)
        compare_cfg(cfg, pretrained_cfg, "otformer.rum.depth", strict=False)
        compare_cfg(cfg, pretrained_cfg, "otformer.rum.num_samples", strict=False)

    # Copy over all PE/SE configs
    for key in cfg.keys():
        if key.startswith("posenc_"):
            cfg[key] = pretrained_cfg[key]

    # Copy over GT config
    cfg.gt = pretrained_cfg.gt

    # Copy over GNN cfg but not those for the prediction head
    compare_cfg(cfg, pretrained_cfg, "gnn.head")
    compare_cfg(cfg, pretrained_cfg, "gnn.layers_post_mp")
    compare_cfg(cfg, pretrained_cfg, "gnn.act", strict=True)
    compare_cfg(cfg, pretrained_cfg, "gnn.dropout")
    head = cfg.gnn.head
    post_mp = cfg.gnn.layers_post_mp
    act = cfg.gnn.act
    drp = cfg.gnn.dropout
    cfg.gnn = pretrained_cfg.gnn
    cfg.gnn.head = head
    cfg.gnn.layers_post_mp = post_mp
    cfg.gnn.act = act
    cfg.gnn.dropout = drp
    return cfg


def init_model_from_pretrained(
    model,
    pretrained_dir,
    freeze_main=False,
    reset_prediction_head=True,
    seed=0,
    weights_path="",
):
    """Copy model parameters from pretrained model except the prediction head.

    Args:
        model: Initialized model with random weights.
        pretrained_dir: Root directory of saved pretrained experiment.
        freeze_main: If True, do not finetune the loaded pretrained parameters
            of the ``main body`` (train the prediction head only), else train all.
        reset_prediction_head: If True, reset parameters of the prediction head,
            else keep the pretrained weights.
        seed: Optionally, the training seed for default ckpt resolution.
        weights_path: If set, load weights from this exact .ckpt file path,
            bypassing the default ``{pretrained_dir}/{seed}/ckpt/`` resolution.

    Returns:
        Updated pytorch model object.
    """
    from torch_geometric.graphgym.checkpoint import MODEL_STATE

    if weights_path:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")
        ckpt_file = weights_path
        logging.info(f"[*] Loading pretrained weights from: {ckpt_file}")
    else:
        ckpt_file = get_final_pretrained_ckpt(
            osp.join(pretrained_dir, str(seed), "ckpt")
        )
        logging.info(f"[*] Loading pretrained weights from: {ckpt_file}")

    ckpt = torch.load(ckpt_file, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and MODEL_STATE in ckpt:
        pretrained_dict = ckpt[MODEL_STATE]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        pretrained_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        pretrained_dict = ckpt
    else:
        raise ValueError(
            f"Unsupported checkpoint format in '{ckpt_file}': type={type(ckpt)}"
        )
    model_dict = model.state_dict()

    def _is_prediction_head_key(name):
        return (
            name == "post_mp"
            or name.startswith("post_mp.")
            or name.startswith("model.post_mp.")
            or ".post_mp." in name
        )

    def _is_encoder_key(name):
        return (
            name == "encoder"
            or name.startswith("encoder.")
            or name.startswith("model.encoder.")
            or ".encoder." in name
        )

    if not list(pretrained_dict.keys())[0].startswith("model."):
        for k in list(pretrained_dict.keys()):
            pretrained_dict[f"model.{k}"] = pretrained_dict.pop(k)

    # Ignore encoder weights for cross-dataset finetuning transfer.
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if not _is_encoder_key(k)
    }
    if reset_prediction_head:
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if not _is_prediction_head_key(k)
        }

    target_keys = [
        k
        for k in model_dict.keys()
        if (
            (not _is_encoder_key(k))
            and ((not _is_prediction_head_key(k)) if reset_prediction_head else True)
        )
    ]
    matched_keys = []
    shape_mismatch = []
    missing_in_ckpt = []
    for k in target_keys:
        if k not in pretrained_dict:
            missing_in_ckpt.append(k)
            continue
        if pretrained_dict[k].shape != model_dict[k].shape:
            shape_mismatch.append(
                (k, tuple(pretrained_dict[k].shape), tuple(model_dict[k].shape))
            )
            continue
        matched_keys.append(k)

    filtered_pretrained_dict = {k: pretrained_dict[k] for k in matched_keys}
    model_dict.update(filtered_pretrained_dict)

    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
    if missing_keys:
        logging.warning(
            f"[*] Missing keys in checkpoint (will be randomly initialized): "
            f"{missing_keys}"
        )
    if unexpected_keys:
        logging.warning(
            f"[*] Unexpected keys in checkpoint (will be ignored): "
            f"{unexpected_keys}"
        )

    loaded_count = len(filtered_pretrained_dict)
    total_target = max(1, len(target_keys))
    strict_match_ratio = (len(matched_keys) / total_target) * 100.0
    logging.info(
        f"[*] Loaded {loaded_count} / {len(model_dict)} parameter groups "
        f"from {osp.basename(ckpt_file)}"
    )
    logging.info("[*] Pretrained transfer policy: encoder parameters are ignored.")
    if shape_mismatch:
        preview = ", ".join([item[0] for item in shape_mismatch[:5]])
        logging.warning(
            "[W] Skipped %d pretrained parameters due to shape mismatch. "
            "Examples: %s",
            len(shape_mismatch),
            preview,
        )
    model._pretrained_load_report = {
        "ckpt_file": ckpt_file,
        "strict_match_ratio": strict_match_ratio,
        "matched": len(matched_keys),
        "target": len(target_keys),
        "missing_in_ckpt": missing_in_ckpt,
        "shape_mismatch": shape_mismatch,
    }
    if strict_match_ratio == 100.0:
        logging.info(
            "[*] Pretrained load check: SUCCESS. Strict parameter match = "
            "100.00%% (%d/%d).",
            len(matched_keys),
            len(target_keys),
        )
    else:
        logging.warning(
            "[W] Pretrained load check: strict parameter match = %.2f%% (%d/%d). "
            "Missing: %d, shape mismatch: %d.",
            strict_match_ratio,
            len(matched_keys),
            len(target_keys),
            len(missing_in_ckpt),
            len(shape_mismatch),
        )

    if freeze_main:
        for key, param in model.named_parameters():
            if not _is_prediction_head_key(key):
                param.requires_grad = False
    return model
