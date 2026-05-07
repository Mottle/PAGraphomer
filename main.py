import datetime
import os
from typing import Optional
import torch
import logging

# PyTorch 2.6 changed torch.load default weights_only to True.
# This breaks PyG/OGB dataset loading which uses custom classes.
# Monkey-patch to restore backward compatibility.
_original_torch_load = torch.load


def _patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(
        f, map_location=map_location, pickle_module=pickle_module, **kwargs
    )


torch.load = _patched_torch_load

import gps  # noqa, register custom modules
from gps.agg_runs import agg_runs
from gps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    set_cfg,
    load_cfg,
    makedirs_rm_exist,
)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import (
    create_optimizer,
    create_scheduler,
    OptimizerConfig,
)
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from gps.finetuning import (
    find_latest_pretrained_dir,
    find_latest_zinc_pretrained_dir,
    init_model_from_pretrained,
    load_pretrained_model_cfg,
)
from gps.logger import create_logger

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def new_optimizer_config(cfg):
    from dataclasses import dataclass

    @dataclass
    class _OptimizerConfig:
        optimizer: str = cfg.optim.optimizer
        base_lr: float = cfg.optim.base_lr
        weight_decay: float = cfg.optim.weight_decay
        momentum: float = cfg.optim.momentum
        muon_momentum: float = getattr(cfg.optim, "muon_momentum", 0.95)
        muon_adjust_lr_fn: Optional[str] = getattr(cfg.optim, "muon_adjust_lr_fn", None)

    return _OptimizerConfig()


def new_scheduler_config(cfg):
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


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError(
                "Running multiple repeats of multiple "
                "splits in one run is not supported."
            )
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == "__main__":
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.pretrained.dir or cfg.pretrained.weights_path:
            cfg = load_pretrained_model_cfg(cfg)
        elif cfg.otformer.finetune.enable and not cfg.pretrained.dir:
            zinc_dir = find_latest_zinc_pretrained_dir(model_type="OTFormerModel")
            if zinc_dir:
                logging.info(
                    f"[*] Auto-selected default ZINC pretrained model: {zinc_dir}. "
                    f"Set pretrained.dir/pretrained.weights_path to override."
                )
                cfg.pretrained.dir = zinc_dir
                cfg = load_pretrained_model_cfg(cfg)
            else:
                auto_dir = find_latest_pretrained_dir(model_type="OTFormerModel")
                if auto_dir:
                    logging.warning(
                        "[W] No ZINC pretraining run found under 'results/'. "
                        "Latest OTFormer pretraining is: %s",
                        auto_dir,
                    )
                raise FileNotFoundError(
                    "OTFormer finetuning requires pretrained weights, but no "
                    "ZINC OTFormer pretraining was found in 'results/'. "
                    "Set 'pretrained.dir' or 'pretrained.weights_path' manually."
                )
        elif cfg.gps.finetune.enable and not cfg.pretrained.dir:
            zinc_dir = find_latest_zinc_pretrained_dir(model_type="GPSModel")
            if zinc_dir:
                logging.info(
                    f"[*] Auto-selected default ZINC pretrained model: {zinc_dir}. "
                    f"Set pretrained.dir/pretrained.weights_path to override."
                )
                cfg.pretrained.dir = zinc_dir
                cfg = load_pretrained_model_cfg(cfg)
            else:
                auto_dir = find_latest_pretrained_dir(model_type="GPSModel")
                if auto_dir:
                    logging.warning(
                        "[W] No ZINC pretraining run found under 'results/'. "
                        "Latest GPS pretraining is: %s",
                        auto_dir,
                    )
                raise FileNotFoundError(
                    "GPS finetuning requires pretrained weights, but no "
                    "ZINC GPS pretraining was found in 'results/'. "
                    "Set 'pretrained.dir' or 'pretrained.weights_path' manually."
                )
        logging.info(
            f"[*] Run ID {run_id}: seed={cfg.seed}, "
            f"split_index={cfg.dataset.split_index}"
        )
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loaders = create_loader()
        # GraphGym's set_dataset_info uses torch.unique(dataset._data.y)
        # which breaks for multi-task datasets with NaN labels (MUV, Tox21,
        # SIDER, Clintox) — each NaN counts as a unique value.
        # Restore correct dim_out and task_type BEFORE create_logger()+create_model().
        if loaders:
            ds = loaders[0].dataset
            if hasattr(ds, "dataset"):
                ds = ds.dataset
            num_tasks = getattr(ds, "num_tasks", 1)
            if num_tasks > 1:
                cfg.share.dim_out = num_tasks
                if cfg.dataset.task_type == "classification":
                    cfg.dataset.task_type = "classification_multilabel"
        loggers = create_logger()
        # GraphGym's create_model reduces dim_out=2 to 1 for classification
        # (binary heuristic, breaks multi-task datasets like Clintox/MUV/SIDER/Tox21).
        # Temporarily override task_type to bypass the heuristic.
        saved_task = cfg.dataset.task_type
        if saved_task == "classification" and cfg.share.dim_out > 2:
            cfg.dataset.task_type = "classification_multitask"
        elif saved_task == "classification" and cfg.share.dim_out == 2 and loaders:
            # dim_out==2 with classification could be single-task binary
            # (unique labels [0,1]) or multi-task with 2 tasks (Clintox).
            # Check num_tasks to distinguish them.
            ds_ref = loaders[0].dataset
            if hasattr(ds_ref, "dataset"):
                ds_ref = ds_ref.dataset
            if hasattr(ds_ref, "num_tasks") and ds_ref.num_tasks > 1:
                cfg.dataset.task_type = "classification_multitask"
        model = create_model()
        cfg.dataset.task_type = saved_task
        if cfg.pretrained.dir or cfg.pretrained.weights_path:
            model = init_model_from_pretrained(
                model,
                cfg.pretrained.dir,
                cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head,
                cfg.pretrained.load_encoder,
                seed=cfg.seed,
                weights_path=cfg.pretrained.weights_path,
            )
            report = getattr(model, "_pretrained_load_report", None)
            if report is not None:
                if report["strict_match_ratio"] == 100.0:
                    logging.info(
                        "[*] Startup check: pretrained weights loaded successfully "
                        "with 100%% strict match (%d/%d).",
                        report["matched"],
                        report["target"],
                    )
                else:
                    logging.warning(
                        "[W] Startup check: pretrained weights loaded, but strict "
                        "match is %.2f%% (%d/%d).",
                        report["strict_match_ratio"],
                        report["matched"],
                        report["target"],
                    )
        elif cfg.otformer.finetune.enable:
            raise ValueError(
                "OTFormer finetuning is enabled but pretrained weights are not "
                "configured. Set 'pretrained.dir' or 'pretrained.weights_path'."
            )
        elif cfg.gps.finetune.enable:
            raise ValueError(
                "GPS finetuning is enabled but pretrained weights are not "
                "configured. Set 'pretrained.dir' or 'pretrained.weights_path'."
            )
        optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info("Num parameters: %s", cfg.params)
        # Start training
        if cfg.train.mode == "standard":
            if cfg.wandb.use:
                logging.warning(
                    "[W] WandB logging is not supported with the "
                    "default train.mode, set it to `custom`"
                )
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)
    # Aggregate results from different seeds
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f"{args.cfg_file}_done")
    logging.info(f"[*] All done: {datetime.datetime.now()}")
