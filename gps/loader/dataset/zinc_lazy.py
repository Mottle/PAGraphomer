import logging
import os
import os.path as osp
import gc
from functools import partial

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import ZINC
from tqdm import tqdm


class ZINCLazyDataset(Dataset):
    """On-disk ZINC dataset — loads graphs individually from .pt files.

    Preprocessing: processes one split at a time in memory (fast),
    then saves each graph individually to disk. Loading: lazy per-graph.
    """

    def __init__(
        self,
        root: str,
        name: str = "full",
        extra_transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.name = name
        self._split_files = []
        self.extra_transform = extra_transform
        super().__init__(root, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["train.pickle", "val.pickle", "test.pickle"]

    @property
    def processed_file_names(self):
        return ["index.pt", "meta.pt", "train_graph_index.pt"]

    def download(self):
        pass

    def _preprocess_split(self, ds, split_name, processed_dir):
        """Preprocess a single ZINC split and save graphs individually."""
        import numpy as np

        from gps.transform.task_preprocessing import task_specific_preprocessing
        from gps.transform.posenc_stats import compute_posenc_stats
        from gps.transform.transforms import pre_transform_in_memory
        from gps.loader.rdkit_features import RDKitFeatureComputer
        from torch_geometric.graphgym.config import cfg
        from torch_geometric.data import Data as PyGData

        mani_pretrain = getattr(getattr(cfg, "srf_rum_mani", None), "pretrain", None)
        is_pretrain = mani_pretrain is not None and getattr(
            mani_pretrain, "enable", False
        )

        phase2 = is_pretrain and (
            (
                getattr(mani_pretrain, "atom_context", None) is not None
                and mani_pretrain.atom_context.get("enable", False)
            )
            or (
                getattr(mani_pretrain, "mol_property", None) is not None
                and mani_pretrain.mol_property.get("enable", False)
            )
        )
        phase3 = is_pretrain and (
            (
                getattr(mani_pretrain, "fingerprint_contrastive", None) is not None
                and mani_pretrain.fingerprint_contrastive.get("enable", False)
            )
            or (
                getattr(mani_pretrain, "scaffold_contrastive", None) is not None
                and mani_pretrain.scaffold_contrastive.get("enable", False)
            )
        )

        # Step 1: task preprocessing (SPD) — batch in-memory
        logging.info(f"  Running task_preprocessing for {split_name}...")
        task_func = partial(task_specific_preprocessing, cfg=cfg)
        data_list = [
            task_func(ds.get(i))
            for i in tqdm(range(len(ds)), desc=f"  SPD {split_name}")
        ]

        # Step 2: RDKit features — compute per graph, fast
        if phase2 or phase3:
            logging.info(f"  Computing RDKit features for {split_name}...")
            computer = RDKitFeatureComputer()
            for data in tqdm(data_list, desc=f"  RDKit {split_name}"):
                if phase2:
                    atom_ctx, mol_props = computer.compute_for_data(data)
                    if atom_ctx is not None:
                        data.atom_context = torch.from_numpy(
                            np.asarray(atom_ctx, dtype=np.float32)
                        )
                    if mol_props is not None:
                        data.mol_property = torch.from_numpy(
                            np.asarray(mol_props, dtype=np.float32)
                        )
                if phase3:
                    mol_fp, scaff_fp = computer.compute_fingerprints(data)
                    if mol_fp is not None:
                        data.mol_fp = torch.from_numpy(
                            np.asarray(mol_fp, dtype=np.float32)
                        )
                    if scaff_fp is not None:
                        data.scaff_fp = torch.from_numpy(
                            np.asarray(scaff_fp, dtype=np.float32)
                        )

        # Step 3: RRWP — compute per graph
        logging.info(f"  Computing RRWP for {split_name}...")
        for data in tqdm(data_list, desc=f"  RRWP {split_name}"):
            compute_posenc_stats(
                data,
                pe_types=["RRWP"],
                is_undirected=True,
                cfg=cfg,
            )

        # Step 4: Save individual files
        graph_files = []
        for i, data in enumerate(tqdm(data_list, desc=f"  Save {split_name}")):
            file_index = len(all_graph_files) + i
            fname = f"graph_{file_index:08d}.pt"
            fpath = osp.join(processed_dir, fname)
            torch.save(data, fpath)
            graph_files.append(fpath)

        return graph_files

    def process(self):
        processed_dir = self.processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        subset = self.name == "subset"
        all_graph_files = []
        global_offset = 0

        for split in ["train", "val", "test"]:
            logging.info(f"Preprocessing ZINC {self.name}/train (batch {split})...")
            ds = ZINC(root=self.root, subset=subset, split=split)

            files = self._preprocess_split(ds, split, processed_dir)
            # Re-index files for flat train-only naming
            renamed = []
            for fname in files:
                new_name = f"graph_{global_offset:08d}.pt"
                os.rename(fname, osp.join(processed_dir, new_name))
                renamed.append(osp.join(processed_dir, new_name))
                global_offset += 1
            all_graph_files.extend(renamed)

            del ds
            gc.collect()

        meta = {"num_graphs": len(all_graph_files), "name": self.name}
        torch.save(meta, osp.join(processed_dir, "meta.pt"))
        torch.save(all_graph_files, osp.join(processed_dir, "index.pt"))

        n_train = len(all_graph_files)
        train_graph_index = torch.arange(n_train, dtype=torch.long)
        val_graph_index = torch.tensor([], dtype=torch.long)
        test_graph_index = torch.tensor([], dtype=torch.long)
        torch.save(
            {
                "train_graph_index": train_graph_index,
                "val_graph_index": val_graph_index,
                "test_graph_index": test_graph_index,
            },
            osp.join(processed_dir, "train_graph_index.pt"),
        )

        logging.info(f"ZINCLazyDataset processed {n_train} graphs (all train)")

    def len(self) -> int:
        if self._split_files:
            return len(self._split_files)
        index_path = osp.join(self.processed_dir, "index.pt")
        if osp.exists(index_path):
            self._split_files = torch.load(index_path, weights_only=False)
            return len(self._split_files)
        return 0

    def get(self, idx: int) -> Data:
        if not self._split_files:
            index_path = osp.join(self.processed_dir, "index.pt")
            self._split_files = torch.load(index_path, weights_only=False)
        data = torch.load(self._split_files[idx], weights_only=False)
        return data

    @property
    def data(self):
        index_path = osp.join(self.processed_dir, "train_graph_index.pt")
        if osp.exists(index_path):
            return torch.load(index_path, weights_only=False)
        return {}
