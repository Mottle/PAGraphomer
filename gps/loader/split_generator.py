import json
import logging
import os
import csv
import gzip
import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import index2mask, set_dataset_attr

from gps.loader.dataset.motil_molecule_csv import MotiLMoleculeCSVDataset


def prepare_splits(dataset):
    """Ready train/val/test splits.

    Determine the type of split from the config and call the corresponding
    split generation / verification function.
    """
    split_mode = cfg.dataset.split_mode

    if split_mode == "standard":
        setup_standard_split(dataset)
    elif split_mode == "scaffold_balanced":
        setup_scaffold_balanced_split(dataset)
    elif split_mode == "motil_scafford_balance":
        setup_motil_scafford_balance_split(dataset)
    elif split_mode == "molmcl_scaffold":
        setup_molmcl_scaffold_split(dataset)
    elif split_mode == "bmol":
        setup_bmol_scaffold_split(dataset)
    elif split_mode == "deepchem_scaffold":
        setup_deepchem_scaffold_split(dataset)
    elif split_mode == "random":
        setup_random_split(dataset)
    elif split_mode.startswith("cv-"):
        cv_type, k = split_mode.split("-")[1:]
        setup_cv_split(dataset, cv_type, int(k))
    elif split_mode == "fixed":
        setup_fixed_split(dataset)
    elif split_mode == "sliced":
        setup_sliced_split(dataset)
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")


def _split_seed():
    split_seed = getattr(cfg.dataset, "split_seed", -1)
    return cfg.seed if split_seed is None or int(split_seed) < 0 else int(split_seed)


def _resolve_split_sizes(split_sizes, dataset_len):
    if len(split_sizes) != 3:
        raise ValueError(
            f"Three split ratios are expected for train/val/test, received "
            f"{len(split_sizes)} split ratios: {repr(split_sizes)}"
        )

    split_sum = sum(split_sizes)
    if split_sum == 1:
        train_size = int(split_sizes[0] * dataset_len)
        val_size = int(split_sizes[1] * dataset_len)
        test_size = int(split_sizes[2] * dataset_len)
    elif split_sum == dataset_len:
        train_size = int(split_sizes[0])
        val_size = int(split_sizes[1])
        test_size = int(split_sizes[2])
    else:
        raise ValueError(
            "The train/val/test split ratios must sum to 1 or dataset length, "
            f"got sum={split_sum} with split={repr(split_sizes)}"
        )

    if train_size + val_size + test_size > dataset_len:
        raise ValueError(
            "Resolved train/val/test split sizes exceed dataset length: "
            f"{train_size + val_size + test_size} > {dataset_len}"
        )

    return train_size, val_size, test_size


def _load_ogbg_mol_smiles(dataset):
    if not hasattr(dataset, "name") or not str(dataset.name).startswith("ogbg-mol"):
        raise ValueError(
            "scaffold_balanced split is currently supported only for ogbg-mol* datasets"
        )

    root_dir = getattr(dataset, "root", None)
    if root_dir is None:
        raise ValueError("Dataset has no root directory; cannot locate raw smiles file")

    candidate_paths = [
        os.path.join(root_dir, "mapping", "mol.csv.gz"),
        os.path.join(root_dir, "raw", "mapping", "mol.csv.gz"),
        os.path.join(root_dir, "raw", "mol.csv.gz"),
    ]

    smiles_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            smiles_path = path
            break

    if smiles_path is None:
        raise FileNotFoundError(
            "Cannot find OGB smiles mapping file. Expected one of: "
            f"{candidate_paths}"
        )

    smiles_list = []
    with gzip.open(smiles_path, mode="rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "smiles" in row:
                smiles_list.append(row["smiles"])
            else:
                first_key = next(iter(row.keys()))
                smiles_list.append(row[first_key])

    if len(smiles_list) != len(dataset):
        raise ValueError(
            "Loaded smiles count does not match dataset length: "
            f"{len(smiles_list)} vs {len(dataset)}"
        )

    return smiles_list


def _load_motil_csv_smiles(dataset):
    if isinstance(dataset, MotiLMoleculeCSVDataset):
        return [dataset[i].smiles for i in range(len(dataset))]

    csv_path = getattr(cfg.dataset, "external_smiles_csv", "")
    if not csv_path:
        dataset_name = getattr(dataset, "name", "")
        csv_name = f"{str(dataset_name).replace('ogbg-mol', '')}.csv"
        csv_path = os.path.join("datasets", "motil_micromolecule", csv_name)

    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.getcwd(), csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MotiL CSV file not found: {csv_path}")

    smiles_list = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                smiles_list.append(row[0])

    if len(smiles_list) != len(dataset):
        if hasattr(dataset, "name") and str(dataset.name).startswith("ogbg-mol"):
            ref_smiles = _load_ogbg_mol_smiles(dataset)
            ref_counter = Counter(ref_smiles)
            aligned_smiles = []
            dropped_smiles = []

            for smiles in smiles_list:
                if ref_counter[smiles] > 0:
                    aligned_smiles.append(smiles)
                    ref_counter[smiles] -= 1
                else:
                    dropped_smiles.append(smiles)

            aligned_counter = Counter(aligned_smiles)
            missing_smiles = []
            for smiles in ref_smiles:
                if aligned_counter[smiles] > 0:
                    aligned_counter[smiles] -= 1
                else:
                    missing_smiles.append(smiles)

            if aligned_smiles == ref_smiles:
                preview = ", ".join(dropped_smiles[:3]) if dropped_smiles else "n/a"
                logging.warning(
                    "Aligned external MotiL CSV smiles to OGB dataset order for %s; "
                    "dropped %d unmatched CSV rows. Examples: %s",
                    getattr(dataset, "name", "dataset"),
                    len(dropped_smiles),
                    preview,
                )
                return aligned_smiles

            if missing_smiles:
                missing_preview = ", ".join(missing_smiles[:3])
            else:
                missing_preview = "n/a"
            raise ValueError(
                "MotiL CSV smiles could not be aligned to OGB dataset order: "
                f"csv={len(smiles_list)} dataset={len(dataset)} path={csv_path}. "
                f"Dropped={len(dropped_smiles)}, missing={len(missing_smiles)}. "
                f"Missing examples: {missing_preview}"
            )

        raise ValueError(
            "MotiL CSV smiles count does not match dataset length: "
            f"{len(smiles_list)} vs {len(dataset)} for {csv_path}"
        )

    return smiles_list


def _safe_scaffold_from_smiles(
    smiles, idx, chem_module, murcko_module, include_chirality=False
):
    mol = chem_module.MolFromSmiles(smiles)
    if mol is None:
        mol = chem_module.MolFromSmiles(smiles, sanitize=False)

    if mol is None:
        return f"__invalid_scaffold_{idx}", True

    try:
        scaffold = murcko_module.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
    except Exception:
        return f"__invalid_scaffold_{idx}", True

    return scaffold, False


def setup_scaffold_balanced_split(dataset):
    """Generate scaffold-balanced train/val/test splits for OGB molecular datasets.

    Mirrors Chemprop's scaffold_balanced strategy used by MotiL:
    1) Group molecules by Bemis-Murcko scaffold.
    2) Separate large scaffold groups (larger than half val/test target size).
    3) Shuffle groups with dataset.split_seed (fallback cfg.seed) and allocate groups to train, then val, then test.
    """
    if cfg.dataset.task != "graph":
        raise ValueError("scaffold_balanced split supports graph-level datasets only")

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as e:
        logging.warning(
            "RDKit is unavailable, cannot compute seed-controlled scaffold_balanced split. "
            "Falling back to standard split. "
            "Install RDKit (e.g. `pixi add rdkit -e default`) to enable MotiL-style splitting. "
            "Original import error: %s",
            repr(e),
        )
        setup_standard_split(dataset)
        return

    split_sizes = cfg.dataset.split
    train_size, val_size, test_size = _resolve_split_sizes(split_sizes, len(dataset))

    smiles_list = _load_ogbg_mol_smiles(dataset)

    scaffold_to_indices = defaultdict(list)
    invalid_smiles = []
    for idx, smiles in enumerate(smiles_list):
        scaffold, is_invalid = _safe_scaffold_from_smiles(
            smiles, idx, Chem, MurckoScaffold
        )
        if is_invalid:
            invalid_smiles.append((idx, smiles))

        scaffold_to_indices[scaffold].append(idx)

    if invalid_smiles:
        preview = ", ".join([f"{i}:{s}" for i, s in invalid_smiles[:3]])
        logging.warning(
            "scaffold_balanced split encountered %d invalid/unscaffoldable SMILES; "
            "assigning each to its own scaffold bucket. Examples: %s",
            len(invalid_smiles),
            preview,
        )

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)

    random.seed(_split_seed())
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    train_index = []
    val_index = []
    test_index = []
    for index_set in index_sets:
        if len(train_index) + len(index_set) <= train_size:
            train_index.extend(index_set)
        elif len(val_index) + len(index_set) <= val_size:
            val_index.extend(index_set)
        else:
            test_index.extend(index_set)

    logging.info(
        "Using scaffold_balanced split for %s with seed=%d: train=%d, val=%d, test=%d",
        getattr(dataset, "name", "dataset"),
        _split_seed(),
        len(train_index),
        len(val_index),
        len(test_index),
    )
    set_dataset_splits(dataset, [train_index, val_index, test_index])


def setup_motil_scafford_balance_split(dataset):
    """MotiL-style scaffold balanced split using copied MotiL CSV source files.

    This matches MotiL more closely by deriving scaffold groups from the copied
    CSV files under `datasets/motil_micromolecule/`, instead of OGB mapping files.
    The split policy itself remains Chemprop's balanced scaffold split.
    """
    if cfg.dataset.task != "graph":
        raise ValueError(
            "motil_scafford_balance split supports graph-level datasets only"
        )

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as e:
        logging.warning(
            "RDKit is unavailable, cannot compute motil_scafford_balance split. "
            "Falling back to standard split. Original import error: %s",
            repr(e),
        )
        setup_standard_split(dataset)
        return

    split_sizes = cfg.dataset.split
    train_size, val_size, test_size = _resolve_split_sizes(split_sizes, len(dataset))
    smiles_list = _load_motil_csv_smiles(dataset)

    scaffold_to_indices = defaultdict(list)
    invalid_smiles = []
    for idx, smiles in enumerate(smiles_list):
        scaffold, is_invalid = _safe_scaffold_from_smiles(
            smiles, idx, Chem, MurckoScaffold
        )
        if is_invalid:
            invalid_smiles.append((idx, smiles))
        scaffold_to_indices[scaffold].append(idx)

    if invalid_smiles:
        preview = ", ".join([f"{i}:{s}" for i, s in invalid_smiles[:3]])
        logging.warning(
            "motil_scafford_balance encountered %d invalid/unscaffoldable SMILES; "
            "assigning each to its own scaffold bucket. Examples: %s",
            len(invalid_smiles),
            preview,
        )

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)

    random.seed(_split_seed())
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    train_index = []
    val_index = []
    test_index = []
    for index_set in index_sets:
        if len(train_index) + len(index_set) <= train_size:
            train_index.extend(index_set)
        elif len(val_index) + len(index_set) <= val_size:
            val_index.extend(index_set)
        else:
            test_index.extend(index_set)

    logging.info(
        "Using motil_scafford_balance split for %s with seed=%d: train=%d, val=%d, test=%d",
        getattr(dataset, "name", "dataset"),
        _split_seed(),
        len(train_index),
        len(val_index),
        len(test_index),
    )
    set_dataset_splits(dataset, [train_index, val_index, test_index])


def _load_molmcl_data_smiles(dataset):
    """Load SMILES from MolMCL's CSV data files under datasets/molmcl_data/.

    Falls back to OGB mapping if the MolMCL CSV is unavailable or its length
    does not match the dataset.
    """
    name = getattr(dataset, "name", "")
    csv_name = name.replace("ogbg-mol", "") + ".csv"
    csv_path = os.path.join("datasets", "molmcl_data", csv_name)
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.getcwd(), csv_path)

    if not os.path.exists(csv_path):
        logging.info(
            "MolMCL CSV not found at %s, falling back to OGB mapping.", csv_path
        )
        return _load_ogbg_mol_smiles(dataset)

    smiles_list = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = None
            if "smiles" in row:
                smiles = row["smiles"]
            elif "SMILES" in row:
                smiles = row["SMILES"]
            elif "mol" in row:
                smiles = row["mol"]
            if smiles is None:
                for k, v in row.items():
                    kl = k.strip().lower()
                    if kl in ("smiles", "mol"):
                        smiles = v
                        break
            if smiles is None and row:
                first_key = next(iter(row.keys()))
                first_val = row[first_key]
                if any(c in first_val for c in "C=NOPF") and len(first_val) > 3:
                    smiles = first_val
            if smiles:
                smiles_list.append(smiles.strip())

    if len(smiles_list) == len(dataset):
        logging.info("Loaded %d SMILES from MolMCL CSV for %s.", len(smiles_list), name)
        return smiles_list

    if len(smiles_list) > len(dataset):
        logging.warning(
            "MolMCL CSV (%d) > dataset (%d) for %s, using first %d entries.",
            len(smiles_list),
            len(dataset),
            name,
            len(dataset),
        )
        return smiles_list[: len(dataset)]

    logging.warning(
        "MolMCL CSV length (%d) < dataset length (%d) for %s, "
        "falling back to OGB mapping.",
        len(smiles_list),
        len(dataset),
        name,
    )
    return _load_ogbg_mol_smiles(dataset)


def setup_molmcl_scaffold_split(dataset):
    """MolMCL-style scaffold split with `balanced=False`.

    Exactly reproduces the split strategy from MolMCL (Wan et al., 2025):
      - Group molecules by Bemis-Murcko scaffold.
      - Sort scaffolds descending by size, resolving ties by the smallest
        index in each group (deterministic).
      - Fill train, then val, then test sequentially using strict ``>``
        cutoffs, matching the MolMCL implementation.
    """
    if cfg.dataset.task != "graph":
        raise ValueError("molmcl_scaffold split supports graph-level datasets only")

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as e:
        logging.warning(
            "RDKit is unavailable, cannot compute molmcl_scaffold split. "
            "Falling back to standard split. Original import error: %s",
            repr(e),
        )
        setup_standard_split(dataset)
        return

    split_sizes = cfg.dataset.split
    train_size, val_size, test_size = _resolve_split_sizes(split_sizes, len(dataset))
    smiles_list = _load_molmcl_data_smiles(dataset)

    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smiles, includeChirality=True
            )
        except Exception:
            continue
        scaffold_to_indices[scaffold].append(idx)

    # MolMCL: sort indices within each scaffold, then sort by size desc
    all_scaffolds = {key: sorted(value) for key, value in scaffold_to_indices.items()}
    scaffold_sets = [
        scaffold_set
        for scaffold, scaffold_set in sorted(
            all_scaffolds.items(),
            key=lambda item: (len(item[1]), item[1][0]),
            reverse=True,
        )
    ]

    # MolMCL uses strict > cutoff (not <= train_size)
    train_cutoff = train_size
    valid_cutoff = train_size + val_size
    train_idx, val_idx, test_idx = [], [], []

    random.seed(_split_seed())
    # MolMCL does NOT shuffle the scaffold order when balanced=False.
    # It simply iterates in the sorted order.
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(val_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                val_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    logging.info(
        "Using molmcl_scaffold split for %s with seed=%d: train=%d, val=%d, test=%d",
        getattr(dataset, "name", "dataset"),
        _split_seed(),
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    set_dataset_splits(dataset, [train_idx, val_idx, test_idx])


def setup_bmol_scaffold_split(dataset):
    """Old-style molmcl scaffold split (bmol) — kept as baseline.

    Uses includeChirality=False, assigns invalid SMILES to their own
    scaffold bucket, and sorts buckets by (len, first_unsorted_index).
    """
    if cfg.dataset.task != "graph":
        raise ValueError("bmol scaffold split supports graph-level datasets only")

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as e:
        logging.warning(
            "RDKit unavailable, falling back to standard split. %s", repr(e)
        )
        setup_standard_split(dataset)
        return

    split_sizes = cfg.dataset.split
    train_size, val_size, test_size = _resolve_split_sizes(split_sizes, len(dataset))
    smiles_list = _load_molmcl_data_smiles(dataset)

    scaffold_to_indices = defaultdict(list)
    invalid_smiles = []
    for idx, smiles in enumerate(smiles_list):
        scaffold, is_invalid = _safe_scaffold_from_smiles(
            smiles, idx, Chem, MurckoScaffold
        )
        if is_invalid:
            invalid_smiles.append((idx, smiles))
        scaffold_to_indices[scaffold].append(idx)

    if invalid_smiles:
        preview = ", ".join([f"{i}:{s}" for i, s in invalid_smiles[:3]])
        logging.warning(
            "bmol scaffold split encountered %d invalid/unscaffoldable "
            "SMILES; assigning each to its own scaffold bucket. Examples: %s",
            len(invalid_smiles),
            preview,
        )

    scaffold_sets = [
        sorted(indices)
        for scaffold, indices in sorted(
            scaffold_to_indices.items(),
            key=lambda item: (len(item[1]), item[1][0]),
            reverse=True,
        )
    ]

    train_cutoff = train_size
    valid_cutoff = train_size + val_size
    train_idx, val_idx, test_idx = [], [], []

    random.seed(_split_seed())
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(val_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                val_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    logging.info(
        "Using bmol scaffold split for %s with seed=%d: train=%d, val=%d, test=%d",
        getattr(dataset, "name", "dataset"),
        _split_seed(),
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    set_dataset_splits(dataset, [train_idx, val_idx, test_idx])


def setup_deepchem_scaffold_split(dataset):
    """DeepChem / Mole-BERT style scaffold split with `balanced=False`.

    Reproduces the scaffold split from Mole-BERT (Xia et al., ICLR 2023),
    which follows the original DeepChem scaffold splitter:
      - Group molecules by Bemis-Murcko scaffold (includeChirality=True).
      - Sort scaffolds descending by size, resolving ties by the smallest
        index in each group (deterministic).
      - Fill train, then val, then test sequentially using strict ``>``
        cutoffs, matching the DeepChem implementation.
    -SMILES are loaded from OGB mapping files (order matches OGB dataset).
    """
    if cfg.dataset.task != "graph":
        raise ValueError("deepchem_scaffold split supports graph-level datasets only")

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as e:
        logging.warning(
            "RDKit is unavailable, cannot compute deepchem_scaffold split. "
            "Falling back to standard split. Original import error: %s",
            repr(e),
        )
        setup_standard_split(dataset)
        return

    split_sizes = cfg.dataset.split
    train_size, val_size, test_size = _resolve_split_sizes(split_sizes, len(dataset))
    smiles_list = _load_ogbg_mol_smiles(dataset)

    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smiles, includeChirality=True
            )
        except Exception:
            continue
        scaffold_to_indices[scaffold].append(idx)

    # DeepChem: sort indices within each scaffold, then sort by size desc
    all_scaffolds = {key: sorted(value) for key, value in scaffold_to_indices.items()}
    scaffold_sets = [
        scaffold_set
        for scaffold, scaffold_set in sorted(
            all_scaffolds.items(),
            key=lambda item: (len(item[1]), item[1][0]),
            reverse=True,
        )
    ]

    # DeepChem uses strict > cutoff (not <=)
    train_cutoff = train_size
    valid_cutoff = train_size + val_size
    train_idx, val_idx, test_idx = [], [], []

    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(val_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                val_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    logging.info(
        "Using deepchem_scaffold split for %s with seed=%d: "
        "train=%d, val=%d, test=%d",
        getattr(dataset, "name", "dataset"),
        _split_seed(),
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    set_dataset_splits(dataset, [train_idx, val_idx, test_idx])


def setup_standard_split(dataset):
    """Select a standard split.

    Use standard splits that come with the dataset. Pick one split based on the
    ``split_index`` from the config file if multiple splits are available.

    GNNBenchmarkDatasets have splits that are not prespecified as masks. Therefore,
    they are handled differently and are first processed to generate the masks.

    Raises:
        ValueError: If any one of train/val/test mask is missing.
        IndexError: If the ``split_index`` is greater or equal to the total
            number of splits available.
    """
    split_index = cfg.dataset.split_index
    task_level = cfg.dataset.task

    if task_level == "node":
        for split_name in "train_mask", "val_mask", "test_mask":
            mask = getattr(dataset.data, split_name, None)
            # Check if the train/val/test split mask is available
            if mask is None:
                raise ValueError(f"Missing '{split_name}' for standard split")

            # Pick a specific split if multiple splits are available
            if mask.dim() == 2:
                if split_index >= mask.shape[1]:
                    raise IndexError(
                        f"Specified split index ({split_index}) is "
                        f"out of range of the number of available "
                        f"splits ({mask.shape[1]}) for {split_name}"
                    )
                set_dataset_attr(
                    dataset, split_name, mask[:, split_index], len(mask[:, split_index])
                )
            else:
                if split_index != 0:
                    raise IndexError(f"This dataset has single standard split")

    elif task_level == "graph":
        for split_name in "train_graph_index", "val_graph_index", "test_graph_index":
            if not hasattr(dataset.data, split_name):
                raise ValueError(f"Missing '{split_name}' for standard split")
        if split_index != 0:
            raise NotImplementedError(
                f"Multiple standard splits not supported "
                f"for dataset task level: {task_level}"
            )

    elif task_level == "link_pred":
        for split_name in "train_edge_index", "val_edge_index", "test_edge_index":
            if not hasattr(dataset.data, split_name):
                raise ValueError(f"Missing '{split_name}' for standard split")
        if split_index != 0:
            raise NotImplementedError(
                f"Multiple standard splits not supported "
                f"for dataset task level: {task_level}"
            )

    else:
        if split_index != 0:
            raise NotImplementedError(
                f"Multiple standard splits not supported "
                f"for dataset task level: {task_level}"
            )


def setup_random_split(dataset):
    """Generate random splits.

    Generate random train/val/test based on the ratios defined in the config
    file.

    Raises:
        ValueError: If the number split ratios is not equal to 3, or the ratios
            do not sum up to 1.
    """
    split_ratios = cfg.dataset.split

    if len(split_ratios) != 3:
        raise ValueError(
            f"Three split ratios is expected for train/val/test, received "
            f"{len(split_ratios)} split ratios: {repr(split_ratios)}"
        )
    elif sum(split_ratios) != 1 and sum(split_ratios) != len(dataset):
        raise ValueError(
            f"The train/val/test split ratios must sum up to 1/length of the dataset, input ratios "
            f"sum up to {sum(split_ratios):.2f} instead: {repr(split_ratios)}"
        )

    dataset_len = len(dataset)
    train_size, val_size, test_size = _resolve_split_sizes(split_ratios, dataset_len)

    if train_size == dataset_len and val_size == 0 and test_size == 0:
        train_index = np.arange(dataset_len)
        val_index = np.array([], dtype=np.int64)
        test_index = np.array([], dtype=np.int64)
        set_dataset_splits(dataset, [train_index, val_index, test_index])
        return

    train_index, val_test_index = next(
        ShuffleSplit(train_size=split_ratios[0], random_state=_split_seed()).split(
            dataset.data.y, dataset.data.y
        )
    )

    if len(val_test_index) == 0:
        val_index = np.array([], dtype=np.int64)
        test_index = np.array([], dtype=np.int64)
        set_dataset_splits(dataset, [train_index, val_index, test_index])
        return

    if isinstance(split_ratios[0], float):
        val_test_ratio = split_ratios[1] / (1 - split_ratios[0])
    else:
        val_test_ratio = split_ratios[1]

    if val_size == 0 and test_size == 0:
        val_index = np.array([], dtype=np.int64)
        test_index = np.array([], dtype=np.int64)
    elif val_size == 0:
        val_index = np.array([], dtype=np.int64)
        test_index = val_test_index
    elif test_size == 0:
        val_index = val_test_index
        test_index = np.array([], dtype=np.int64)
    else:
        val_index, test_index = next(
            ShuffleSplit(train_size=val_test_ratio, random_state=_split_seed()).split(
                dataset.data.y[val_test_index], dataset.data.y[val_test_index]
            )
        )
        val_index = val_test_index[val_index]
        test_index = val_test_index[test_index]

    set_dataset_splits(dataset, [train_index, val_index, test_index])


def setup_fixed_split(dataset):
    """Generate fixed splits.

    Generate fixed train/val/test based on the ratios defined in the config
    file.
    """
    train_index = list(range(cfg.dataset.split[0]))
    val_index = list(range(cfg.dataset.split[0], sum(cfg.dataset.split[:2])))
    test_index = list(range(sum(cfg.dataset.split[:2]), sum(cfg.dataset.split)))

    set_dataset_splits(dataset, [train_index, val_index, test_index])


def setup_sliced_split(dataset):
    """Generate sliced splits.

    Generate sliced train/val/test based on the ratios defined in the config
    file.
    """
    train_index = list(range(*cfg.dataset.split[0]))
    val_index = list(range(*cfg.dataset.split[1]))
    test_index = list(range(*cfg.dataset.split[2]))

    set_dataset_splits(dataset, [train_index, val_index, test_index])


def set_dataset_splits(dataset, splits):
    """Set given splits to the dataset object.

    Args:
        dataset: PyG dataset object
        splits: List of train/val/test split indices

    Raises:
        ValueError: If any pair of splits has intersecting indices
    """
    # First check whether splits intersect and raise error if so.
    for i in range(len(splits) - 1):
        for j in range(i + 1, len(splits)):
            n_intersect = len(set(splits[i]) & set(splits[j]))
            if n_intersect != 0:
                raise ValueError(
                    f"Splits must not have intersecting indices: "
                    f"split #{i} (n = {len(splits[i])}) and "
                    f"split #{j} (n = {len(splits[j])}) have "
                    f"{n_intersect} intersecting indices"
                )

    task_level = cfg.dataset.task
    if task_level == "node":
        split_names = ["train_mask", "val_mask", "test_mask"]
        for split_name, split_index in zip(split_names, splits):
            mask = index2mask(split_index, size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_name, mask, len(mask))

    elif task_level == "graph":
        split_names = ["train_graph_index", "val_graph_index", "test_graph_index"]
        for split_name, split_index in zip(split_names, splits):
            set_dataset_attr(dataset, split_name, split_index, len(split_index))

    else:
        raise ValueError(f"Unsupported dataset task level: {task_level}")


def setup_cv_split(dataset, cv_type, k):
    """Generate cross-validation splits.

    Generate `k` folds for cross-validation based on `cv_type` procedure. Save
    these to disk or load existing splits, then select particular train/val/test
    split based on cfg.dataset.split_index from the config object.

    Args:
        dataset: PyG dataset object
        cv_type: Identifier for which sklearn fold splitter to use
        k: how many cross-validation folds to split the dataset into

    Raises:
        IndexError: If the `split_index` is greater than or equal to `k`
    """
    split_index = cfg.dataset.split_index
    split_dir = cfg.dataset.split_dir

    if split_index >= k:
        raise IndexError(
            f"Specified split_index={split_index} is "
            f"out of range of the number of folds k={k}"
        )

    os.makedirs(split_dir, exist_ok=True)
    save_file = os.path.join(
        split_dir, f"{cfg.dataset.format}_{dataset.name}_{cv_type}-{k}.json"
    )
    if not os.path.isfile(save_file):
        create_cv_splits(dataset, cv_type, k, save_file)
    with open(save_file) as f:
        cv = json.load(f)
    assert cv["dataset"] == dataset.name, "Unexpected dataset CV splits"
    assert cv["n_samples"] == len(dataset), "Dataset length does not match"
    assert cv["n_splits"] > split_index, "Fold selection out of range"
    assert k == cv["n_splits"], f"Expected k={k}, but {cv['n_splits']} found"

    test_ids = cv[str(split_index)]
    val_ids = cv[str((split_index + 1) % k)]
    train_ids = []
    for i in range(k):
        if i != split_index and i != (split_index + 1) % k:
            train_ids.extend(cv[str(i)])

    set_dataset_splits(dataset, [train_ids, val_ids, test_ids])


def create_cv_splits(dataset, cv_type, k, file_name):
    """Create cross-validation splits and save them to file."""
    n_samples = len(dataset)
    if cv_type == "stratifiedkfold":
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=123)
        kf_split = kf.split(np.zeros(n_samples), dataset.data.y)
    elif cv_type == "kfold":
        kf = KFold(n_splits=k, shuffle=True, random_state=123)
        kf_split = kf.split(np.zeros(n_samples))
    else:
        ValueError(f"Unexpected cross-validation type: {cv_type}")

    splits = {
        "n_samples": n_samples,
        "n_splits": k,
        "cross_validator": kf.__str__(),
        "dataset": dataset.name,
    }
    for i, (_, ids) in enumerate(kf_split):
        splits[i] = ids.tolist()
    with open(file_name, "w") as f:
        json.dump(splits, f)
    logging.info(f"[*] Saved newly generated CV splits by {kf} to {file_name}")
