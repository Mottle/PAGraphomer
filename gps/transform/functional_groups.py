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
        data.fg_type = torch.empty((0,), dtype=torch.long)
        data.fg_atom_fg_id = torch.empty((0,), dtype=torch.long)
        data.fg_ptr = torch.tensor([0], dtype=torch.long)
        data.fg_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.fg_edge_attr = torch.empty((0, 1), dtype=torch.long)
        data.fg_edge_ptr = torch.tensor([0], dtype=torch.long)
        data.fg_edge_fg_id = torch.empty((0,), dtype=torch.long)
        data.fg_edge_src_pos = torch.empty((0,), dtype=torch.long)
        data.fg_edge_dst_pos = torch.empty((0,), dtype=torch.long)
        data.fg_count = torch.tensor([0], dtype=torch.long)
        return data

    _, fg_smarts = _load_motil_functional_group_rules()
    atom_index = []
    type_index_per_atom = []
    fg_id_per_atom = []
    type_index_per_fg = []
    fg_ptr = [0]
    fg_edge_index = []
    fg_edge_attr = []
    fg_edge_ptr = [0]
    fg_edge_fg_id = []
    fg_edge_src_pos = []
    fg_edge_dst_pos = []
    fg_id = 0
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    for fg_type, smarts in enumerate(fg_smarts):
        matches = mol.GetSubstructMatches(smarts)
        if len(matches) > 0 and len(matches[0]) > 1:
            match = list(matches[0])
            node_set = set(match)
            local_pos = {node_id: pos for pos, node_id in enumerate(match)}
            atom_index.extend(match)
            type_index_per_atom.extend([fg_type] * len(match))
            fg_id_per_atom.extend([fg_id] * len(match))
            type_index_per_fg.append(fg_type)

            if edge_index is not None:
                for eid in range(edge_index.size(1)):
                    src = int(edge_index[0, eid].item())
                    dst = int(edge_index[1, eid].item())
                    if src in node_set and dst in node_set:
                        fg_edge_index.append((src, dst))
                        fg_edge_fg_id.append(fg_id)
                        fg_edge_src_pos.append(local_pos[src])
                        fg_edge_dst_pos.append(local_pos[dst])
                        if edge_attr is not None:
                            fg_edge_attr.append(edge_attr[eid])

            fg_id += 1
            fg_ptr.append(len(atom_index))
            fg_edge_ptr.append(len(fg_edge_index))

    data.fg_atom_index = torch.tensor(atom_index, dtype=torch.long).view(-1)
    data.fg_type_index = torch.tensor(type_index_per_atom, dtype=torch.long).view(-1)
    data.fg_type = torch.tensor(type_index_per_fg, dtype=torch.long).view(-1)
    data.fg_atom_fg_id = torch.tensor(fg_id_per_atom, dtype=torch.long).view(-1)
    data.fg_ptr = torch.tensor(fg_ptr, dtype=torch.long).view(-1)
    if fg_edge_index:
        src, dst = zip(*fg_edge_index)
        data.fg_edge_index = torch.tensor([src, dst], dtype=torch.long)
        if fg_edge_attr:
            data.fg_edge_attr = torch.stack(fg_edge_attr).to(torch.long)
        else:
            data.fg_edge_attr = torch.empty((0, 1), dtype=torch.long)
    else:
        data.fg_edge_index = torch.empty((2, 0), dtype=torch.long)
        if edge_attr is not None and edge_attr.dim() > 1:
            data.fg_edge_attr = torch.empty((0, edge_attr.size(-1)), dtype=torch.long)
        else:
            data.fg_edge_attr = torch.empty((0, 1), dtype=torch.long)
    data.fg_edge_ptr = torch.tensor(fg_edge_ptr, dtype=torch.long).view(-1)
    data.fg_edge_fg_id = torch.tensor(fg_edge_fg_id, dtype=torch.long).view(-1)
    data.fg_edge_src_pos = torch.tensor(fg_edge_src_pos, dtype=torch.long).view(-1)
    data.fg_edge_dst_pos = torch.tensor(fg_edge_dst_pos, dtype=torch.long).view(-1)
    data.fg_count = torch.tensor([fg_id], dtype=torch.long)
    return data
