from torch_geometric.graphgym.register import register_config


@register_config("split")
def set_cfg_split(cfg):
    """Reconfigure the default config value for dataset split options.

    Returns:
        Reconfigured split configuration use by the experiment.
    """

    # Default to selecting the standard split that ships with the dataset
    cfg.dataset.split_mode = "standard"

    # Choose a particular split to use if multiple splits are available
    cfg.dataset.split_index = 0

    # Random seed used specifically for dataset split generation.
    # If left unset/negative, fall back to the global cfg.seed.
    cfg.dataset.split_seed = -1

    # Dir to cache cross-validation splits
    cfg.dataset.split_dir = "./splits"

    # Choose to run multiple splits in one program execution, if set,
    # takes the precedence over cfg.dataset.split_index for split selection
    cfg.run_multiple_splits = []
