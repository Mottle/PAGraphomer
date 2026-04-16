import logging
import os
import time
from glob import glob

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch


def _save_epoch_weights_rolling(model, epoch):
    if not bool(getattr(cfg.grit.pretrain, "save_epoch_weights", True)):
        return
    keep_n = max(1, int(getattr(cfg.grit.pretrain, "keep_last_epoch_weights", 3)))
    save_dir = os.path.join(cfg.run_dir, "pretrain_weights")
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, f"grit_epoch_{int(epoch):04d}.pt")
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    all_ckpts = sorted(glob(os.path.join(save_dir, "grit_epoch_*.pt")))
    if len(all_ckpts) > keep_n:
        for path in all_ckpts[: len(all_ckpts) - keep_n]:
            os.remove(path)


def _masked_accuracy(atom_logits, atom_target, atom_mask):
    if not atom_mask.any():
        return 0.0
    pred = atom_logits[atom_mask].argmax(dim=-1)
    return (pred == atom_target[atom_mask]).float().mean().item()


def _masked_edge_accuracy(edge_logits, edge_target, edge_supervision_idx):
    if (
        edge_logits is None
        or edge_target is None
        or edge_supervision_idx is None
        or edge_supervision_idx.numel() == 0
    ):
        return 0.0
    pred = edge_logits[edge_supervision_idx].argmax(dim=-1)
    return (pred == edge_target[edge_supervision_idx]).float().mean().item()


def _train_epoch_pretrain(
    logger, loader, model, optimizer, scheduler, batch_accumulation
):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for step, batch in enumerate(loader):
        batch.split = "train"
        batch.to(torch.device(cfg.accelerator))
        pred, true = model(batch)
        if not hasattr(batch, "gps_aux") or "loss" not in batch.gps_aux:
            raise RuntimeError(
                "GRIT pretraining requires `batch.gps_aux['loss']`. "
                "Set `grit.pretrain.enable=True` and use GritTransformer."
            )
        loss = batch.gps_aux["loss"]
        losses = batch.gps_aux.get("losses", {})
        atom_logits = batch.gps_aux["atom_logits"]
        atom_target = batch.gps_aux["atom_target"]
        atom_mask = batch.gps_aux["atom_mask"]
        mask_acc = _masked_accuracy(atom_logits, atom_target, atom_mask)
        edge_logits = batch.gps_aux.get("edge_logits")
        edge_target = batch.gps_aux.get("edge_target")
        edge_supervision_idx = batch.gps_aux.get("edge_supervision_idx")
        edge_acc = _masked_edge_accuracy(edge_logits, edge_target, edge_supervision_idx)
        loss.backward()

        if ((step + 1) % batch_accumulation == 0) or (step + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.clip_grad_norm_value
                )
            optimizer.step()
            optimizer.zero_grad()

        proxy_true = torch.zeros_like(true.detach())
        proxy_pred = torch.full_like(true.detach(), loss.detach().item())
        logger.update_stats(
            true=proxy_true.to("cpu", non_blocking=True),
            pred=proxy_pred.to("cpu", non_blocking=True),
            loss=loss.detach().cpu().item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
            pre_mask_atom=losses.get("mask_atom", loss).detach().cpu().item(),
            pre_mask_acc=mask_acc,
            pre_mask_nodes=int(atom_mask.sum().item()),
            pre_mask_edge=losses.get("mask_edge", loss.new_zeros(()))
            .detach()
            .cpu()
            .item(),
            pre_edge_acc=edge_acc,
            pre_mask_edges=(
                0 if edge_supervision_idx is None else int(edge_supervision_idx.numel())
            ),
        )
        time_start = time.time()


@torch.no_grad()
def _eval_epoch_pretrain(logger, loader, model, split="val"):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        pred, true = model(batch)
        if hasattr(batch, "gps_aux") and "loss" in batch.gps_aux:
            loss = batch.gps_aux["loss"]
            losses = batch.gps_aux.get("losses", {})
            atom_logits = batch.gps_aux["atom_logits"]
            atom_target = batch.gps_aux["atom_target"]
            atom_mask = batch.gps_aux["atom_mask"]
            edge_logits = batch.gps_aux.get("edge_logits")
            edge_target = batch.gps_aux.get("edge_target")
            edge_supervision_idx = batch.gps_aux.get("edge_supervision_idx")
            extra = {
                "pre_mask_atom": losses.get("mask_atom", loss).detach().cpu().item(),
                "pre_mask_acc": _masked_accuracy(atom_logits, atom_target, atom_mask),
                "pre_mask_nodes": int(atom_mask.sum().item()),
                "pre_mask_edge": losses.get("mask_edge", loss.new_zeros(()))
                .detach()
                .cpu()
                .item(),
                "pre_edge_acc": _masked_edge_accuracy(
                    edge_logits, edge_target, edge_supervision_idx
                ),
                "pre_mask_edges": (
                    0
                    if edge_supervision_idx is None
                    else int(edge_supervision_idx.numel())
                ),
            }
        else:
            loss = torch.tensor(0.0, device=batch.x.device)
            extra = {}
        proxy_true = torch.zeros_like(true.detach())
        proxy_pred = torch.full_like(true.detach(), loss.detach().item())
        logger.update_stats(
            true=proxy_true.to("cpu", non_blocking=True),
            pred=proxy_pred.to("cpu", non_blocking=True),
            loss=loss.detach().cpu().item(),
            lr=0.0,
            time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
            **extra,
        )
        time_start = time.time()


@register_train("grit_pretrain")
def grit_pretrain_train(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)

    use_eval_splits = (
        bool(getattr(cfg.grit.pretrain, "eval_splits", False)) and len(loggers) > 1
    )
    active_loggers = list(loggers) if use_eval_splits else [loggers[0]]
    active_loaders = list(loaders) if use_eval_splits else [loaders[0]]

    num_splits = len(active_loggers)
    split_names = ["val", "test"]
    perf = [[] for _ in range(num_splits)]
    epoch_times = []

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        t0 = time.perf_counter()
        _train_epoch_pretrain(
            active_loggers[0],
            active_loaders[0],
            model,
            optimizer,
            scheduler,
            cfg.optim.batch_accumulation,
        )
        perf[0].append(active_loggers[0].write_epoch(cur_epoch))

        if use_eval_splits and is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                _eval_epoch_pretrain(
                    active_loggers[i],
                    active_loaders[i],
                    model,
                    split=split_names[i - 1],
                )
                perf[i].append(active_loggers[i].write_epoch(cur_epoch))
        elif use_eval_splits:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1] if perf[i] else perf[0][-1])

        if cfg.optim.scheduler == "reduce_on_plateau":
            if use_eval_splits and num_splits > 1:
                scheduler.step(perf[1][-1]["loss"])
            else:
                scheduler.step(perf[0][-1]["loss"])
        else:
            scheduler.step()

        epoch_times.append(time.perf_counter() - t0)
        _save_epoch_weights_rolling(model, cur_epoch)
        if (
            cfg.train.enable_ckpt
            and not cfg.train.ckpt_best
            and is_ckpt_epoch(cur_epoch)
        ):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if is_eval_epoch(cur_epoch):
            if use_eval_splits and num_splits > 1:
                best_epoch = np.array([vp["loss"] for vp in perf[1]]).argmin()
                if (
                    cfg.train.enable_ckpt
                    and cfg.train.ckpt_best
                    and best_epoch == cur_epoch
                ):
                    save_ckpt(model, optimizer, scheduler, cur_epoch)
                    if cfg.train.ckpt_clean:
                        clean_ckpt()
                logging.info(
                    f"> Epoch {cur_epoch}: "
                    f"train_loss={perf[0][-1]['loss']:.4f} "
                    f"val_loss={perf[1][-1]['loss']:.4f} "
                    f"test_loss={perf[2][-1]['loss']:.4f}"
                )
            else:
                if cfg.train.enable_ckpt and cfg.train.ckpt_best:
                    best_epoch = np.array([vp["loss"] for vp in perf[0]]).argmin()
                    if best_epoch == cur_epoch:
                        save_ckpt(model, optimizer, scheduler, cur_epoch)
                        if cfg.train.ckpt_clean:
                            clean_ckpt()
                logging.info(
                    f"> Epoch {cur_epoch}: train_loss={perf[0][-1]['loss']:.4f}"
                )

    logging.info(f"Avg time per epoch: {np.mean(epoch_times):.2f}s")
    for logger in active_loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    logging.info("GRIT pretraining done, results saved in %s", cfg.run_dir)
