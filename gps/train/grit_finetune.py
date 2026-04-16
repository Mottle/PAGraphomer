import logging
import time

import numpy as np
from torch_geometric.graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch

from gps.train.otformer_finetune import (
    _reset_optimizer_param_groups,
    _set_param_group_lr,
    _split_param_groups,
    eval_epoch_finetune,
    train_epoch_finetune,
)


@register_train("grit_finetune")
def grit_finetune_train(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info("Checkpoint found, Task already done")
    else:
        logging.info("Start from epoch %s", start_epoch)

    freeze_backbone = getattr(cfg.grit.finetune, "freeze_backbone", False)
    freeze_backbone_epochs = int(
        getattr(cfg.grit.finetune, "freeze_backbone_epochs", 0)
    )
    backbone_lr_ratio = getattr(cfg.grit.finetune, "backbone_lr_ratio", 0.1)
    encoder_lr_ratio = getattr(cfg.grit.finetune, "encoder_lr_ratio", 1.0)
    train_encoder_when_freeze = bool(
        getattr(cfg.grit.finetune, "train_encoder_when_freeze_backbone", True)
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

        if (
            cfg.train.enable_ckpt
            and not cfg.train.ckpt_best
            and is_ckpt_epoch(cur_epoch)
        ):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp["loss"] for vp in val_perf]).argmin()
            if cfg.metric_best != "auto":
                best_epoch = getattr(
                    np.array([vp[cfg.metric_best] for vp in val_perf]), cfg.metric_agg
                )()

            if (
                cfg.train.enable_ckpt
                and cfg.train.ckpt_best
                and best_epoch == cur_epoch
            ):
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:
                    clean_ckpt()

            logging.info(
                "> Epoch %d: took %.1fs | Best so far: epoch %d | "
                "train_loss: %.4f | val_loss: %.4f | test_loss: %.4f",
                cur_epoch,
                time.perf_counter() - start_time,
                best_epoch,
                perf[0][best_epoch]["loss"],
                perf[1][best_epoch]["loss"],
                perf[2][best_epoch]["loss"],
            )

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    logging.info("GRIT finetuning done, results saved in %s", cfg.run_dir)
