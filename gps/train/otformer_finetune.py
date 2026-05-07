import logging
import os
import time

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from gps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from gps.utils import cfg_to_dict, flatten_dict, make_wandb_name


def _save_weights(last_state, best_val_state=None, best_test_state=None):
    weight_dir = os.path.join(cfg.run_dir, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    torch.save(last_state, os.path.join(weight_dir, "last.pt"))
    logging.info(f"[*] Saved last weights to {os.path.join(weight_dir, 'last.pt')}")
    if best_val_state is not None:
        torch.save(best_val_state, os.path.join(weight_dir, "bestValTest.pt"))
        logging.info(
            f"[*] Saved bestValTest weights to {os.path.join(weight_dir, 'bestValTest.pt')}"
        )
    if best_test_state is not None:
        torch.save(best_test_state, os.path.join(weight_dir, "bestTest.pt"))
        logging.info(
            f"[*] Saved bestTest weights to {os.path.join(weight_dir, 'bestTest.pt')}"
        )


def _is_prediction_head_param(name):
    return (
        name == "post_mp"
        or name.startswith("post_mp.")
        or name.startswith("model.post_mp.")
        or ".post_mp." in name
    )


def _is_encoder_param(name):
    return (
        name == "encoder"
        or name.startswith("encoder.")
        or name.startswith("model.encoder.")
        or ".encoder." in name
    )


def _split_param_groups(model):
    head_params = []
    encoder_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if _is_prediction_head_param(name):
            head_params.append(param)
        elif _is_encoder_param(name):
            encoder_params.append(param)
        else:
            backbone_params.append(param)
    return head_params, encoder_params, backbone_params


def _reset_optimizer_param_groups(optimizer, param_groups):
    optimizer.param_groups[0]["params"] = param_groups[0]["params"]
    optimizer.param_groups[0]["lr"] = param_groups[0]["lr"]
    optimizer.param_groups[0]["group_name"] = param_groups[0]["group_name"]
    while len(optimizer.param_groups) > 1:
        optimizer.param_groups.pop()
    for group in param_groups[1:]:
        optimizer.add_param_group(group)


def _set_param_group_lr(optimizer, group_name, lr):
    for group in optimizer.param_groups:
        if group.get("group_name") == group_name:
            group["lr"] = lr


def train_epoch_finetune(
    logger, loader, model, optimizer, scheduler, batch_accumulation
):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        batch.split = "train"
        batch.to(torch.device(cfg.accelerator))
        pred, true = model(batch)
        if cfg.dataset.name == "ogbg-code2":
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to("cpu", non_blocking=True)
            _pred = pred_score.detach().to("cpu", non_blocking=True)
        loss.backward()
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.clip_grad_norm_value
                )
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
        )
        time_start = time.time()


@torch.no_grad()
def eval_epoch_finetune(logger, loader, model, split="val"):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        pred, true = model(batch)
        if cfg.dataset.name == "ogbg-code2":
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to("cpu", non_blocking=True)
            _pred = pred_score.detach().to("cpu", non_blocking=True)
        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=0,
            time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
        )
        time_start = time.time()


@register_train("otformer_finetune")
def otformer_finetune_train(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info("Checkpoint found, Task already done")
    else:
        logging.info("Start from epoch %s", start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except ImportError:
            raise ImportError("WandB is not installed.")
        if cfg.wandb.name == "":
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(
            entity=cfg.wandb.entity, project=cfg.wandb.project, name=wandb_name
        )
        run.config.update(cfg_to_dict(cfg))

    freeze_backbone = getattr(cfg.otformer.finetune, "freeze_backbone", False)
    freeze_backbone_epochs = int(
        getattr(cfg.otformer.finetune, "freeze_backbone_epochs", 0)
    )
    backbone_lr_ratio = getattr(cfg.otformer.finetune, "backbone_lr_ratio", 0.1)
    encoder_lr_ratio = getattr(cfg.otformer.finetune, "encoder_lr_ratio", 1.0)
    train_encoder_when_freeze = bool(
        getattr(cfg.otformer.finetune, "train_encoder_when_freeze_backbone", True)
    )
    staged_finetune = freeze_backbone and freeze_backbone_epochs > 0

    head_params, encoder_params, backbone_params = _split_param_groups(model)

    param_groups = [
        {
            "params": head_params,
            "lr": cfg.optim.base_lr,
            "group_name": "head",
        }
    ]
    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": cfg.optim.base_lr * encoder_lr_ratio,
                "group_name": "encoder",
            }
        )
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": cfg.optim.base_lr * backbone_lr_ratio,
                "group_name": "backbone",
            }
        )
    _reset_optimizer_param_groups(optimizer, param_groups)

    if freeze_backbone:
        for param in head_params:
            param.requires_grad = True
        for param in encoder_params:
            param.requires_grad = train_encoder_when_freeze
        for param in backbone_params:
            param.requires_grad = False
        if not train_encoder_when_freeze:
            _set_param_group_lr(optimizer, "encoder", 0.0)
        _set_param_group_lr(optimizer, "backbone", 0.0)

        if staged_finetune:
            logging.info(
                "[*] Stage-1 freeze enabled for %d epochs: training head%s, "
                "backbone frozen.",
                freeze_backbone_epochs,
                " + encoder" if train_encoder_when_freeze else "",
            )
        else:
            logging.info(
                "[*] Backbone frozen for all epochs. Training head%s.",
                " + encoder" if train_encoder_when_freeze else "",
            )
    else:
        for param in head_params:
            param.requires_grad = True
        for param in encoder_params:
            param.requires_grad = True
        for param in backbone_params:
            param.requires_grad = True
        logging.info(
            "[*] Full fine-tuning. Encoder lr ratio: %s, Backbone lr ratio: %s",
            encoder_lr_ratio,
            backbone_lr_ratio,
        )

    num_splits = len(loggers)
    split_names = ["val", "test"]
    perf = [[] for _ in range(num_splits)]
    best_val_state = None
    best_test_state = None

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        if staged_finetune and cur_epoch == freeze_backbone_epochs:
            for param in encoder_params:
                param.requires_grad = True
            for param in backbone_params:
                param.requires_grad = True
            _set_param_group_lr(
                optimizer, "encoder", cfg.optim.base_lr * encoder_lr_ratio
            )
            _set_param_group_lr(
                optimizer, "backbone", cfg.optim.base_lr * backbone_lr_ratio
            )
            logging.info(
                "[*] Stage-2 full fine-tune starts at epoch %d. "
                "Encoder/backbone are now trainable.",
                cur_epoch,
            )

        start_time = time.perf_counter()
        train_epoch_finetune(
            loggers[0],
            loaders[0],
            model,
            optimizer,
            scheduler,
            cfg.optim.batch_accumulation,
        )
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch_finetune(
                    loggers[i], loaders[i], model, split=split_names[i - 1]
                )
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == "reduce_on_plateau":
            scheduler.step(val_perf[-1]["loss"])
        else:
            scheduler.step()

        if cfg.wandb.use:
            wandb.log(flatten_dict(perf[0][-1]), step=cur_epoch)

        if is_eval_epoch(cur_epoch):
            metric_vals_val = [vp[cfg.metric_best] for vp in perf[1]]
            metric_vals_test = [vp[cfg.metric_best] for vp in perf[2]]
            if cfg.metric_agg == "argmax":
                best_val_now = np.argmax(metric_vals_val) == len(metric_vals_val) - 1
                best_test_now = np.argmax(metric_vals_test) == len(metric_vals_test) - 1
            else:
                best_val_now = np.argmin(metric_vals_val) == len(metric_vals_val) - 1
                best_test_now = np.argmin(metric_vals_test) == len(metric_vals_test) - 1
            if best_val_now:
                best_val_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if best_test_now:
                best_test_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        if cfg.train.enable_ckpt and is_ckpt_epoch(cur_epoch):
            if cfg.train.ckpt_best:
                metric_vals = [vp[cfg.metric_best] for vp in val_perf]
                if cfg.metric_agg == "argmax":
                    best_idx = np.argmax(metric_vals)
                else:
                    best_idx = np.argmin(metric_vals)
                if best_idx == len(val_perf) - 1:
                    save_ckpt(model, optimizer, scheduler, cur_epoch)
                    logging.info(f"[*] Saved best ckpt at epoch {cur_epoch}")
            else:
                save_ckpt(model, optimizer, scheduler, cur_epoch)

        if getattr(cfg.train, "clean_ckpt", False):
            clean_ckpt()

        logging.info(
            f"Epoch {cur_epoch}: {perf[0][-1].get('loss', 0):.4f} "
            f"({time.perf_counter() - start_time:.2f}s)"
        )

    best_epoch = np.array([vp["loss"] for vp in val_perf]).argmin()
    if cfg.metric_best != "auto":
        if cfg.metric_agg == "argmax":
            best_epoch = np.array([vp[cfg.metric_best] for vp in val_perf]).argmax()
        else:
            best_epoch = np.array([vp[cfg.metric_best] for vp in val_perf]).argmin()

    best_test_epoch = np.array([vp["loss"] for vp in perf[2]]).argmin()
    if cfg.metric_best != "auto":
        if cfg.metric_agg == "argmax":
            best_test_epoch = np.array([vp[cfg.metric_best] for vp in perf[2]]).argmax()
        else:
            best_test_epoch = np.array([vp[cfg.metric_best] for vp in perf[2]]).argmin()

    last_epoch = len(perf[2]) - 1

    logging.info(f"[*] Best epoch: {best_epoch}")
    logging.info(f"[*] BestValTest: {perf[2][best_epoch]}")
    logging.info(f"[*] LastTest: {perf[2][last_epoch]}")
    logging.info(f"[*] BestTest: {perf[2][best_test_epoch]}")
    for split_name in split_names:
        idx = split_names.index(split_name) + 1
        logging.info(f"[*] Best {split_name}: {perf[idx][best_epoch]}")

    _save_weights(
        last_state=model.state_dict(),
        best_val_state=best_val_state,
        best_test_state=best_test_state,
    )

    if cfg.wandb.use:
        run.finish()
