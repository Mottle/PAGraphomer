import logging
import os
import time
from dataclasses import dataclass
from glob import glob

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch

from gps.optimizer.extra_optimizers import ExtendedSchedulerConfig


def _try_load_epoch_checkpoint(model, optimizer, scheduler):
    try:
        return load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    except Exception as exc:
        logging.warning(
            "Skipping auto-resume checkpoint because it is incompatible with the current config: %s",
            exc,
        )
        return 0


def _should_store_epoch_predictions():
    return not bool(getattr(cfg.grit.motil_pretrain, "enable", False))


def _update_pretrain_logger(
    logger,
    loss,
    lr,
    time_used,
    diffusion_loss,
    contrast_mol_loss,
    contrast_fgs_loss,
):
    logger._iter += 1
    logger._size_current += 1
    logger._loss += loss.detach().cpu().item()
    logger._lr = lr
    logger._params = cfg.params
    logger._time_used += time_used
    logger._time_total += time_used
    for key, val in {
        "pre_diffusion": diffusion_loss.detach().cpu().item(),
        "pre_contrast_mol": contrast_mol_loss.detach().cpu().item(),
        "pre_contrast_fgs": contrast_fgs_loss.detach().cpu().item(),
    }.items():
        if key not in logger._custom_stats:
            logger._custom_stats[key] = val
        else:
            logger._custom_stats[key] += val


def _save_epoch_weights_rolling(model, epoch):
    if not bool(getattr(cfg.grit.motil_pretrain, "save_epoch_weights", True)):
        return
    keep_n = max(1, int(getattr(cfg.grit.motil_pretrain, "keep_last_epoch_weights", 3)))
    save_dir = os.path.join(cfg.run_dir, "pretrain_weights")
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, f"grit_motil_epoch_{int(epoch):04d}.pt")
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    all_ckpts = sorted(glob(os.path.join(save_dir, "grit_motil_epoch_*.pt")))
    if len(all_ckpts) > keep_n:
        for path in all_ckpts[: len(all_ckpts) - keep_n]:
            os.remove(path)


@dataclass
class _OptimizerConfig:
    optimizer: str
    base_lr: float
    weight_decay: float
    momentum: float
    muon_momentum: float
    muon_adjust_lr_fn: str


class _DualOptimizer:
    def __init__(self, diffusion_optimizer, contrast_optimizer):
        self.diffusion = diffusion_optimizer
        self.contrast = contrast_optimizer

    def zero_grad(self, set_to_none=True):
        self.diffusion.zero_grad(set_to_none=set_to_none)
        self.contrast.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "diffusion": self.diffusion.state_dict(),
            "contrast": self.contrast.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.diffusion.load_state_dict(state_dict["diffusion"])
        self.contrast.load_state_dict(state_dict["contrast"])


class _DualScheduler:
    def __init__(self, diffusion_scheduler, contrast_scheduler):
        self.diffusion = diffusion_scheduler
        self.contrast = contrast_scheduler

    def step(self, metric=None):
        if cfg.optim.scheduler == "reduce_on_plateau":
            self.diffusion.step(metric)
            self.contrast.step(metric)
        else:
            self.diffusion.step()
            self.contrast.step()

    def state_dict(self):
        return {
            "diffusion": self.diffusion.state_dict(),
            "contrast": self.contrast.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.diffusion.load_state_dict(state_dict["diffusion"])
        self.contrast.load_state_dict(state_dict["contrast"])

    def get_last_lr(self):
        if hasattr(self.contrast, "get_last_lr"):
            return self.contrast.get_last_lr()
        return [group["lr"] for group in self.contrast.optimizer.param_groups]


def _build_optimizer_config(base_lr):
    return _OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
        muon_momentum=getattr(cfg.optim, "muon_momentum", 0.95),
        muon_adjust_lr_fn=getattr(cfg.optim, "muon_adjust_lr_fn", ""),
    )


def _build_scheduler_config():
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch,
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience,
        min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period,
    )


def _unwrap_graphgym_model(model):
    return getattr(model, "model", model)


def _build_dual_optimizers_and_schedulers(model):
    core_model = _unwrap_graphgym_model(model)

    diffusion_params = list(core_model.encoder.parameters())
    if (
        hasattr(core_model, "motil_diffusion_head")
        and core_model.motil_diffusion_head is not None
    ):
        diffusion_params += list(core_model.motil_diffusion_head.parameters())

    contrast_modules = [core_model.encoder, core_model.layers]
    if hasattr(core_model, "pre_mp"):
        contrast_modules.append(core_model.pre_mp)
    contrast_params = []
    for module in contrast_modules:
        contrast_params.extend(list(module.parameters()))

    diffusion_optimizer = create_optimizer(
        diffusion_params,
        _build_optimizer_config(
            float(getattr(cfg.grit.motil_pretrain, "diffusion_lr", cfg.optim.base_lr))
        ),
    )
    contrast_optimizer = create_optimizer(
        contrast_params,
        _build_optimizer_config(
            float(getattr(cfg.grit.motil_pretrain, "contrast_lr", cfg.optim.base_lr))
        ),
    )

    scheduler_cfg = _build_scheduler_config()
    diffusion_scheduler = create_scheduler(diffusion_optimizer, scheduler_cfg)
    contrast_scheduler = create_scheduler(contrast_optimizer, scheduler_cfg)
    return (
        _DualOptimizer(diffusion_optimizer, contrast_optimizer),
        _DualScheduler(diffusion_scheduler, contrast_scheduler),
        diffusion_optimizer,
        contrast_optimizer,
    )


def _optimizer_parameters(optimizer):
    return [p for group in optimizer.param_groups for p in group["params"]]


def _train_epoch_pretrain(
    logger,
    loader,
    model,
    optimizer,
    scheduler,
    diffusion_optimizer,
    contrast_optimizer,
):
    model.train()
    optimizer.zero_grad()
    accum_steps = max(1, int(getattr(cfg.optim, "batch_accumulation", 1)))
    time_start = time.time()
    cycle_mode = str(
        getattr(cfg.grit.motil_pretrain, "task_cycle", "alternate")
    ).lower()
    contrast_task = "contrast_mol"
    store_epoch_predictions = _should_store_epoch_predictions()
    diffusion_optimizer.zero_grad(set_to_none=True)
    contrast_optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch.split = "train"
        batch.to(torch.device(cfg.accelerator))

        batch.grit_pretrain_phase = "diffusion"
        pred, true = model(batch)
        if not hasattr(batch, "grit_aux") or "loss" not in batch.grit_aux:
            raise RuntimeError(
                "GRIT MotiL-style pretraining requires `batch.grit_aux['loss']`. "
                "Set `grit.motil_pretrain.enable=True` and use GritTransformer."
            )
        diffusion_loss = batch.grit_aux["losses"].get(
            "diffusion", batch.grit_aux["loss"]
        )
        weighted_diffusion_loss = (
            float(getattr(cfg.grit.motil_pretrain, "w_diffusion", 1.0)) * diffusion_loss
        )
        (weighted_diffusion_loss / accum_steps).backward()

        if hasattr(batch, "grit_aux"):
            batch.grit_aux.clear()

        batch.grit_pretrain_phase = "contrast"
        batch.grit_pretrain_task = contrast_task
        pred, true = model(batch)
        if not hasattr(batch, "grit_aux") or "losses" not in batch.grit_aux:
            raise RuntimeError("Missing GRIT MotiL contrast losses in batch.grit_aux")

        losses = batch.grit_aux.get("losses", {})
        contrast_mol_loss = losses.get("contrast_mol", diffusion_loss.new_zeros(()))
        contrast_fgs_loss = losses.get("contrast_fgs", diffusion_loss.new_zeros(()))

        if cycle_mode == "alternate":
            if contrast_task == "contrast_mol":
                contrast_loss = (
                    float(getattr(cfg.grit.motil_pretrain, "w_contrast", 1.0))
                    * contrast_mol_loss
                )
                contrast_task = "contrast_fgs"
            else:
                contrast_loss = (
                    float(getattr(cfg.grit.motil_pretrain, "w_fgs", 1.0))
                    * contrast_fgs_loss
                )
                contrast_task = "contrast_mol"
            loss = contrast_loss
        else:
            loss = (
                float(getattr(cfg.grit.motil_pretrain, "w_contrast", 1.0))
                * contrast_mol_loss
                + float(getattr(cfg.grit.motil_pretrain, "w_fgs", 1.0))
                * contrast_fgs_loss
            )

        (loss / accum_steps).backward()

        should_step = ((step + 1) % accum_steps == 0) or (step + 1 == len(loader))
        if should_step:
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    _optimizer_parameters(diffusion_optimizer),
                    cfg.optim.clip_grad_norm_value,
                )
                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for group in contrast_optimizer.param_groups
                        for p in group["params"]
                    ],
                    cfg.optim.clip_grad_norm_value,
                )
            diffusion_optimizer.step()
            contrast_optimizer.step()
            diffusion_optimizer.zero_grad(set_to_none=True)
            contrast_optimizer.zero_grad(set_to_none=True)

        time_used = time.time() - time_start
        if store_epoch_predictions:
            proxy_true = torch.zeros_like(true.detach())
            proxy_pred = torch.full_like(
                true.detach(),
                (weighted_diffusion_loss.detach() + loss.detach()).item(),
            )
            logger.update_stats(
                true=proxy_true.to("cpu", non_blocking=True),
                pred=proxy_pred.to("cpu", non_blocking=True),
                loss=(weighted_diffusion_loss.detach() + loss.detach()).cpu().item(),
                lr=scheduler.get_last_lr()[0],
                time_used=time_used,
                params=cfg.params,
                dataset_name=cfg.dataset.name,
                pre_diffusion=weighted_diffusion_loss.detach().cpu().item(),
                pre_contrast_mol=contrast_mol_loss.detach().cpu().item(),
                pre_contrast_fgs=contrast_fgs_loss.detach().cpu().item(),
            )
            del proxy_true, proxy_pred
        else:
            _update_pretrain_logger(
                logger,
                loss=weighted_diffusion_loss.detach() + loss.detach(),
                lr=scheduler.get_last_lr()[0],
                time_used=time_used,
                diffusion_loss=weighted_diffusion_loss,
                contrast_mol_loss=contrast_mol_loss,
                contrast_fgs_loss=contrast_fgs_loss,
            )

        if hasattr(batch, "grit_aux"):
            batch.grit_aux.clear()
        if hasattr(batch, "grit_pretrain_task"):
            delattr(batch, "grit_pretrain_task")
        if hasattr(batch, "grit_pretrain_phase"):
            delattr(batch, "grit_pretrain_phase")
        del pred, true, loss, weighted_diffusion_loss
        del diffusion_loss, contrast_mol_loss, contrast_fgs_loss
        del batch
        time_start = time.time()


@torch.no_grad()
def _eval_epoch_pretrain(logger, loader, model, split="val"):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        batch.grit_pretrain_phase = "joint"
        pred, true = model(batch)
        if hasattr(batch, "grit_aux") and "loss" in batch.grit_aux:
            loss = batch.grit_aux["loss"]
            losses = batch.grit_aux.get("losses", {})
            extra = {
                "pre_diffusion": losses.get("diffusion", loss).detach().cpu().item(),
                "pre_contrast_mol": losses.get("contrast_mol", loss)
                .detach()
                .cpu()
                .item(),
                "pre_contrast_fgs": losses.get("contrast_fgs", loss)
                .detach()
                .cpu()
                .item(),
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
        if hasattr(batch, "grit_aux"):
            batch.grit_aux.clear()
        if hasattr(batch, "grit_pretrain_phase"):
            delattr(batch, "grit_pretrain_phase")
        time_start = time.time()


@register_train("grit_motil_pretrain")
def grit_motil_pretrain_train(loggers, loaders, model, optimizer, scheduler):
    optimizer, scheduler, diffusion_optimizer, contrast_optimizer = (
        _build_dual_optimizers_and_schedulers(model)
    )
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = _try_load_epoch_checkpoint(model, optimizer, scheduler)

    use_eval_splits = (
        bool(getattr(cfg.grit.motil_pretrain, "eval_splits", False))
        and len(loggers) > 1
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
            diffusion_optimizer,
            contrast_optimizer,
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
    logging.info("GRIT MotiL-style pretraining done, results saved in %s", cfg.run_dir)
