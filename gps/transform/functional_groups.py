import os
from functools import lru_cache

import torch
from rdkit import Chem
from torch_geometric.utils.smiles import to_rdmol


@lru_cache(maxsize=1)
def _load_motil_functional_group_rules():
    rules_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "resources", "funcgroup.txt"
    )
    names = []
    smarts = []
    with open(rules_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            names.append(parts[0])
            smarts.append(Chem.MolFromSmarts(parts[1]))
    return names, smarts


def add_motil_functional_groups(data):
    mol = None
    if hasattr(data, "smiles") and data.smiles is not None:
        mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        try:
            mol = to_rdmol(data)
        except Exception:
            mol = None
    if mol is None:
        data.fg_atom_index = torch.empty((0,), dtype=torch.long)
        data.fg_type_index = torch.empty((0,), dtype=torch.long)
        data.fg_ptr = torch.tensor([0], dtype=torch.long)
        data.fg_count = torch.tensor([0], dtype=torch.long)
        return data

    _, fg_smarts = _load_motil_functional_group_rules()
    atom_index = []
    type_index = []
    fg_ptr = [0]
    fg_id = 0
    for fg_type, smarts in enumerate(fg_smarts):
        matches = mol.GetSubstructMatches(smarts)
        if len(matches) > 0 and len(matches[0]) > 1:
            match = list(matches[0])
            atom_index.extend(match)
            type_index.extend([fg_type] * len(match))
            fg_id += 1
            fg_ptr.append(len(atom_index))

    data.fg_atom_index = torch.tensor(atom_index, dtype=torch.long).view(-1)
    data.fg_type_index = torch.tensor(type_index, dtype=torch.long).view(-1)
    data.fg_ptr = torch.tensor(fg_ptr, dtype=torch.long).view(-1)
    data.fg_count = torch.tensor([fg_id], dtype=torch.long)
    return data
