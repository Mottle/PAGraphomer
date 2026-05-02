"""CReM-based scaffold-invariant molecular perturbation for MANI pretraining.

Generates perturbed versions of ZINC molecules that preserve the Murcko scaffold
while replacing side chains using chemically reasonable mutations (CReM).
"""

import hashlib
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

# Module-level cache for perturbed Data objects
PERTURBED_CACHE = None  # Dict[int, Optional[Data]], keyed by dataset index


def _get_cache_path(
    dataset_name: str,
    num_train: int,
    db_path: str,
    max_size: int,
    radius: int,
    cache_dir: str = "datasets/crem_cache",
) -> str:
    """Build deterministic cache file path from parameters."""
    db_name = os.path.basename(db_path)
    params = f"{dataset_name}_{num_train}_{db_name}_{max_size}_{radius}"
    h = hashlib.sha256(params.encode()).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"crem_perturbations_{h}.pt")


def load_perturbed_cache(
    dataset_name: str,
    num_train: int,
    db_path: str,
    max_size: int,
    radius: int,
    cache_dir: str = "datasets/crem_cache",
) -> Optional[Dict[int, Optional[Data]]]:
    """Load persistent perturbation cache from disk if it exists.

    Returns:
        Dict mapping dataset index to perturbed Data, or None if cache missing.
    """
    cache_path = _get_cache_path(
        dataset_name, num_train, db_path, max_size, radius, cache_dir
    )
    if not os.path.exists(cache_path):
        return None
    try:
        cache = torch.load(cache_path, map_location="cpu")
        logging.info(
            f"Loaded CReM perturbation cache from {cache_path} ({len(cache)} entries)"
        )
        return cache
    except Exception as e:
        logging.warning(f"Failed to load CReM cache from {cache_path}: {e}")
        return None


def save_perturbed_cache(
    cache: Dict[int, Optional[Data]],
    dataset_name: str,
    num_train: int,
    db_path: str,
    max_size: int,
    radius: int,
    cache_dir: str = "datasets/crem_cache",
) -> None:
    """Save perturbation cache to disk atomically."""
    cache_path = _get_cache_path(
        dataset_name, num_train, db_path, max_size, radius, cache_dir
    )
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = cache_path + ".tmp"
    try:
        torch.save(cache, tmp_path)
        os.replace(tmp_path, cache_path)
        logging.info(
            f"Saved CReM perturbation cache to {cache_path} ({len(cache)} entries)"
        )
    except Exception as e:
        logging.warning(f"Failed to save CReM cache to {cache_path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def scaffold_invariant_perturb(
    mol: Chem.Mol,
    db_name: str,
    num_candidates: int = 1,
    max_size: int = 5,
    radius: int = 2,
    max_replacements: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Chem.Mol]:
    """Generate scaffold-invariant molecular perturbations using CReM.

    Restricts mutations to terminal side chains not part of the Murcko scaffold,
    then filters results to ensure scaffold identity is preserved.

    Args:
        mol: RDKit Mol object
        db_name: path to CReM fragment database
        num_candidates: number of perturbations to return
        max_size: maximum number of heavy atoms in a replacing fragment
        radius: context radius for CReM replacement
        max_replacements: max CReM mutations per molecule (None = all)
        seed: random seed

    Returns:
        List of perturbed RDKit Mol objects
    """
    if seed is not None:
        import random

        random.seed(seed)

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return []

    scaff_smi = Chem.MolToSmiles(scaffold)
    mapped = mol.GetSubstructMatch(scaffold)
    scaffold_atoms = set(mapped)

    replaceable = set(range(mol.GetNumAtoms())) - scaffold_atoms
    if not replaceable:
        return []

    from crem.crem import mutate_mol2
    import copy

    replace_ids = list(replaceable)
    kwa = {"max_size": max_size, "radius": radius, "replace_ids": replace_ids}
    if max_replacements is not None:
        kwa["max_replacements"] = max_replacements

    try:
        mutations = mutate_mol2(mol, db_name=db_name, **kwa)
    except Exception as e:
        logging.warning(f"CReM mutate_mol2 failed: {e}")
        return []

    filtered = []
    for smi in mutations:
        if len(filtered) >= num_candidates:
            break
        try:
            perturbed_mol = Chem.MolFromSmiles(smi)
            if perturbed_mol is None:
                continue
            perturbed_scaff = MurckoScaffold.GetScaffoldForMol(perturbed_mol)
            perturbed_smi = (
                Chem.MolToSmiles(perturbed_scaff)
                if perturbed_scaff.GetNumAtoms() > 0
                else ""
            )
            if perturbed_smi == scaff_smi:
                filtered.append(perturbed_mol)
        except Exception:
            continue

    if seed is not None:
        import random

        random.seed()

    return filtered


def graph_data_from_rdkit_mol(
    mol: Chem.Mol,
    atom_type_map: dict,
    num_atom_types: int = 21,
    num_bond_types: int = 4,
) -> Optional[Data]:
    """Convert an RDKit Mol to a PyG Data object with ZINC atom/bond types.

    Args:
        mol: RDKit Mol
        atom_type_map: mapping from (symbol, formal_charge) to type index
        num_atom_types: max atom type index + 1
        num_bond_types: number of bond types

    Returns:
        PyG Data object or None on failure
    """
    if mol is None:
        return None

    try:
        mol = Chem.AddHs(mol)
    except Exception:
        pass

    n = mol.GetNumAtoms()
    if n == 0:
        return None

    node_types = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        key = (symbol, charge)
        if key in atom_type_map:
            node_types.append(atom_type_map[key])
        else:
            node_types.append(0)

    src, dst, bond_types = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = int(bond.GetBondTypeAsDouble())
        bt = min(bt, num_bond_types - 1)
        src.append(i)
        dst.append(j)
        bond_types.append(bt)
        src.append(j)
        dst.append(i)
        bond_types.append(bt)

    if len(src) == 0:
        return None

    return Data(
        x=torch.tensor(node_types, dtype=torch.long).unsqueeze(-1),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(bond_types, dtype=torch.long),
        num_nodes=n,
    )


def build_atom_type_map() -> dict:
    """Build ZINC atom type reverse mapping (symbol, charge) → type index."""
    from gps.loader.rdkit_features import ZINC_ATOM_TYPE_MAP

    rev = {}
    for t, (sym, chg) in ZINC_ATOM_TYPE_MAP.items():
        rev[(sym, chg)] = t
    return rev


class CReMPerturbationGenerator:
    """Precompute and cache CReM perturbations for ZINC molecules."""

    def __init__(
        self, db_path: str, num_candidates: int = 1, max_size: int = 5, radius: int = 2
    ):
        self.db_path = db_path
        self.num_candidates = num_candidates
        self.max_size = max_size
        self.radius = radius
        self.atom_type_map = build_atom_type_map()
        self._fail_count = 0

    def generate(self, data: Data) -> Optional[Data]:
        """Generate one perturbed PyG Data object from a ZINC graph."""
        from gps.loader.rdkit_features import graph_to_rdkit_mol

        mol = graph_to_rdkit_mol(
            data.x, data.edge_index, data.edge_attr, data.num_nodes
        )
        if mol is None:
            self._fail_count += 1
            return None

        perturbed_mols = scaffold_invariant_perturb(
            mol,
            db_name=self.db_path,
            num_candidates=self.num_candidates,
            max_size=self.max_size,
            radius=self.radius,
        )

        if not perturbed_mols:
            self._fail_count += 1
            return None

        perturbed_data = graph_data_from_rdkit_mol(
            perturbed_mols[0],
            atom_type_map=self.atom_type_map,
        )

        if perturbed_data is None:
            self._fail_count += 1

        return perturbed_data

    def generate_smiles(self, data: Data) -> Optional[str]:
        """Generate one perturbed SMILES string."""
        from gps.loader.rdkit_features import graph_to_rdkit_mol

        mol = graph_to_rdkit_mol(
            data.x, data.edge_index, data.edge_attr, data.num_nodes
        )
        if mol is None:
            self._fail_count += 1
            return None

        perturbed_mols = scaffold_invariant_perturb(
            mol,
            db_name=self.db_path,
            num_candidates=self.num_candidates,
            max_size=self.max_size,
            radius=self.radius,
        )

        if not perturbed_mols:
            self._fail_count += 1
            return None

        try:
            return Chem.MolToSmiles(perturbed_mols[0])
        except Exception:
            self._fail_count += 1
            return None
