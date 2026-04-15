import csv
import os
import os.path as osp

import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data, InMemoryDataset


class MotiLMoleculeCSVDataset(InMemoryDataset):
    """Build molecular graph datasets directly from copied MotiL CSV files.

    The first CSV column must be `smiles`. Remaining columns are treated as
    task targets. Empty target strings are converted to NaN, matching Chemprop's
    missing-label handling for multilabel datasets like Tox21.
    """

    def __init__(
        self,
        root,
        csv_path,
        dataset_name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.csv_path = csv_path
        self.dataset_name = dataset_name
        self.name = dataset_name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [osp.basename(self.csv_path)]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        if not osp.exists(self.csv_path):
            raise FileNotFoundError(f"MotiL CSV file not found: {self.csv_path}")

    def process(self):
        data_list = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            num_tasks = max(1, len(header) - 1)
            lower_header = [col.lower() for col in header[1:]]
            is_binary_single_task = num_tasks == 1 and lower_header == ["class"]
            for row in reader:
                if not row:
                    continue
                smiles = row[0]
                graph = smiles2graph(smiles)

                data = Data()
                data.__num_nodes__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.long)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.long)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.long)
                data.smiles = smiles

                targets = row[1:]
                if is_binary_single_task:
                    value = targets[0] if targets else ""
                    target = -1 if value == "" else int(float(value))
                    data.y = torch.tensor([target], dtype=torch.long)
                elif num_tasks == 1:
                    value = targets[0] if targets else ""
                    target = float("nan") if value == "" else float(value)
                    data.y = torch.tensor([target], dtype=torch.float)
                else:
                    parsed = [float("nan") if v == "" else float(v) for v in targets]
                    data.y = torch.tensor(parsed, dtype=torch.float)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        if not data_list:
            raise ValueError(f"No graphs were constructed from CSV: {self.csv_path}")

        torch.save(self.collate(data_list), self.processed_paths[0])
