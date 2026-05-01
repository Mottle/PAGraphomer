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


def _joint_pretrain_loss(losses):
    return (
        cfg.otformer.pretrain.w_mask_atom * losses["mask_atom"]
        + cfg.otformer.pretrain.w_motif_mask * losses["motif_mask"]
        + cfg.otformer.pretrain.w_edge_denoise * losses["edge_denoise"]
        + cfg.otformer.pretrain.w_ot_prior * losses["ot_prior"]
    )


def _save_epoch_weights_rolling(model, epoch):
    if not bool(getattr(cfg.otformer.pretrain, "save_epoch_weights", True)):
        return
    keep_n = max(1, int(getattr(cfg.otformer.pretrain, "keep_last_epoch_weights", 3)))
    save_dir = os.path.join(cfg.run_dir, "pretrain_weights")
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, f"otformer_epoch_{int(epoch):04d}.pt")
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    all_ckpts = sorted(glob(os.path.join(save_dir, "otformer_epoch_*.pt")))
    if len(all_ckpts) > keep_n:
        for path in all_ckpts[: len(all_ckpts) - keep_n]:
            os.remove(path)


def _load_latest_epoch_weights(model):
    """Resume OTFormer pretraining from latest epoch weight snapshot.

    Priority order for resume is handled by caller:
    1) latest `pretrain_weights/otformer_epoch_*.pt`
    2) GraphGym checkpoint under `ckpt/`

    Returns the next epoch index to run from.
    """
    pretrain_dir = os.path.join(cfg.run_dir, "pretrain_weights")
    weight_paths = sorted(glob(os.path.join(pretrain_dir, "otformer_epoch_*.pt")))
    if not weight_paths:
        return None

    latest_path = weight_paths[-1]
    payload = torch.load(latest_path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload)
    epoch = int(payload.get("epoch", -1))
    model.load_state_dict(state_dict, strict=True)
    logging.info(
        "[*] Resumed OTFormer pretraining model weights from %s (epoch=%d)",
        latest_path,
        epoch,
    )
    return epoch + 1


def _train_epoch_pretrain(
    logger, loader, model, optimizer, scheduler, batch_accumulation, scaler=None
):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for step, batch in enumerate(loader):
        batch.split = "train"
        batch.to(torch.device(cfg.accelerator))
        with torch.amp.autocast("cuda"):
            pred, true = model(batch)
            if not hasattr(batch, "otformer_aux") or "losses" not in batch.otformer_aux:
                raise RuntimeError(
                    "OTFormer pretraining requires `batch.otformer_aux['losses']`. "
                    "Set `otformer.pretrain.enable=True` and use OTFormerModel."
                )
            losses = batch.otformer_aux["losses"]
            loss = _joint_pretrain_loss(losses)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if ((step + 1) % batch_accumulation == 0) or (step + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.clip_grad_norm_value
                )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # For pretraining, drive logger metrics from pretraining objective
        # instead of downstream head prediction.
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
            pre_mask_atom=losses["mask_atom"].detach().cpu().item(),
            pre_motif_mask=losses["motif_mask"].detach().cpu().item(),
            pre_edge_denoise=losses["edge_denoise"].detach().cpu().item(),
            pre_ot_prior=losses["ot_prior"].detach().cpu().item(),
        )
        time_start = time.time()


@torch.no_grad()
def _eval_epoch_pretrain(logger, loader, model, split="val"):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        with torch.amp.autocast("cuda"):
            pred, true = model(batch)
        if hasattr(batch, "otformer_aux") and "losses" in batch.otformer_aux:
            losses = batch.otformer_aux["losses"]
            loss = _joint_pretrain_loss(losses)
            extra = {
                "pre_mask_atom": losses["mask_atom"].detach().cpu().item(),
                "pre_motif_mask": losses["motif_mask"].detach().cpu().item(),
                "pre_edge_denoise": losses["edge_denoise"].detach().cpu().item(),
                "pre_ot_prior": losses["ot_prior"].detach().cpu().item(),
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


@register_train("otformer_pretrain")
def otformer_pretrain_train(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = _load_latest_epoch_weights(model)
        if start_epoch is None:
            start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
            logging.info(
                "[*] No pretrain_weights snapshot found; fell back to GraphGym ckpt resume."
            )

    use_eval_splits = bool(getattr(cfg.otformer.pretrain, "eval_splits", False))
    num_splits = len(loggers)
    split_names = ["val", "test"]
    perf = [[] for _ in range(num_splits)]
    epoch_times = []
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        t0 = time.perf_counter()
        _train_epoch_pretrain(
            loggers[0],
            loaders[0],
            model,
            optimizer,
            scheduler,
            cfg.optim.batch_accumulation,
            scaler=scaler,
        )
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if use_eval_splits and is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                _eval_epoch_pretrain(
                    loggers[i], loaders[i], model, split=split_names[i - 1]
                )
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
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
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    logging.info("OTFormer pretraining done, results saved in %s", cfg.run_dir)
