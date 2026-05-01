"""RDKit-based feature computation for MANI pretraining (Phase 2).

Provides atom-level context features and molecule-level property descriptors
computed on-the-fly during dataset loading.
"""

from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, GraphDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdchem import HybridizationType

# ---------------------------------------------------------------------------
# ZINC atom type mapping (from gps/encoder/type_dict_encoder.py)
# ---------------------------------------------------------------------------

ZINC_ATOM_TYPE_MAP = {
    0: ("C", 0),
    1: ("O", 0),
    2: ("N", 0),
    3: ("F", 0),
    4: ("C", 1),
    5: ("S", 0),
    6: ("Cl", 0),
    7: ("O", -1),
    8: ("N", 1),
    9: ("Br", 0),
    10: ("N", 3),
    11: ("N", 2),
    12: ("N", 1),
    13: ("N", -1),
    14: ("S", -1),
    15: ("I", 0),
    16: ("P", 0),
    17: ("O", 1),
    18: ("N", 0),
    19: ("O", 1),
    20: ("S", 1),
    21: ("P", 1),
    22: ("P", 2),
    23: ("C", -1),
    24: ("P", 1),
    25: ("S", 1),
    26: ("C", -1),
    27: ("P", 1),
}


# ---------------------------------------------------------------------------
# Graph to RDKit Mol conversion
# ---------------------------------------------------------------------------


def graph_to_rdkit_mol(
    atom_types: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_atoms: int,
) -> Optional[Chem.Mol]:
    """Build an RDKit Mol object from a ZINC graph representation.

    Args:
        atom_types: [N, 1] or [N] integer tensor of ZINC atom types
        edge_index: [2, E] edge connectivity
        edge_attr: [E] integer bond types (1=single, 2=double, 3=triple)
        num_atoms: actual number of atoms (to ignore padding)

    Returns:
        RDKit Mol object or None if construction fails
    """
    atom_types = atom_types.squeeze()
    if atom_types.dim() == 0:
        return None

    mol = Chem.RWMol()
    atom_idx_map = {}

    for i in range(num_atoms):
        atype = int(atom_types[i].item())
        if atype not in ZINC_ATOM_TYPE_MAP:
            continue
        symbol, formal_charge = ZINC_ATOM_TYPE_MAP[atype]
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(formal_charge)
        idx = mol.AddAtom(atom)
        atom_idx_map[i] = idx

    if len(atom_idx_map) == 0:
        return None

    added = set()
    for j in range(edge_index.shape[1]):
        src = int(edge_index[0, j].item())
        dst = int(edge_index[1, j].item())
        if src >= num_atoms or dst >= num_atoms:
            continue
        if src not in atom_idx_map or dst not in atom_idx_map:
            continue
        if src == dst:
            continue
        key = tuple(sorted((src, dst)))
        if key in added:
            continue
        added.add(key)

        bond_type_int = int(edge_attr[j].item())
        if bond_type_int == 1:
            bond_type = Chem.BondType.SINGLE
        elif bond_type_int == 2:
            bond_type = Chem.BondType.DOUBLE
        elif bond_type_int == 3:
            bond_type = Chem.BondType.TRIPLE
        else:
            continue

        try:
            mol.AddBond(atom_idx_map[src], atom_idx_map[dst], bond_type)
        except Exception:
            pass

    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
    except Exception:
        mol = mol.GetMol()

    return mol


# ---------------------------------------------------------------------------
# Atom context feature definitions
# ---------------------------------------------------------------------------

ATOM_TYPES = list(range(119))
DEGREES = [0, 1, 2, 3, 4, 5, 6]
HYBRIDIZATIONS = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    HybridizationType.UNSPECIFIED,
    HybridizationType.S,
]
FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]
VALENCES = [0, 1, 2, 3, 4, 5, 6]
RING_SIZES = [3, 4, 5, 6, 7, 8]


def _onek_encoding(value, allowable_set):
    if value not in allowable_set:
        value = allowable_set[-1]
    return [int(v == value) for v in allowable_set]


def atom_context_features(mol: Chem.Mol, atom_idx: int) -> np.ndarray:
    """Compute rich atom context features for a single atom.

    Returns array of shape (158,) with:
    - atomic number (119-dim one-hot)
    - degree (7-dim one-hot)
    - hybridization (7-dim one-hot)
    - formal charge (7-dim one-hot)
    - valence (7-dim one-hot)
    - aromatic, ring, chiral (3-dim bool)
    - mass (1-dim float)
    - ring sizes (6-dim one-hot)
    - radical electrons (1-dim int)
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    feats = []

    feats += _onek_encoding(atom.GetAtomicNum(), ATOM_TYPES)
    feats += _onek_encoding(atom.GetDegree(), DEGREES)
    feats += _onek_encoding(atom.GetHybridization(), HYBRIDIZATIONS)
    feats += _onek_encoding(atom.GetFormalCharge(), FORMAL_CHARGES)
    feats += _onek_encoding(atom.GetTotalValence(), VALENCES)
    feats.append(int(atom.GetIsAromatic()))
    feats.append(int(atom.IsInRing()))
    feats.append(int(atom.HasProp("_ChiralityPossible")))
    feats.append(atom.GetMass() / 12.0)

    ring_sizes = [0] * len(RING_SIZES)
    for ring in mol.GetRingInfo().AtomRings():
        if atom_idx in ring:
            size = len(ring)
            if size in RING_SIZES:
                ring_sizes[RING_SIZES.index(size)] = 1
    feats += ring_sizes

    feats.append(atom.GetNumRadicalElectrons())
    return np.array(feats, dtype=np.float32)


def mol_atom_context_features(mol: Chem.Mol) -> np.ndarray:
    """Compute atom context features for all atoms.

    Returns array of shape (N_atoms, 158).
    """
    num_atoms = mol.GetNumAtoms()
    features = np.zeros((num_atoms, 158), dtype=np.float32)
    for i in range(num_atoms):
        features[i] = atom_context_features(mol, i)
    return features


# ---------------------------------------------------------------------------
# Molecular property descriptors
# ---------------------------------------------------------------------------

PROPERTY_NAMES = [
    "MolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumSaturatedRings",
    "NumAliphaticRings",
    "NumAromaticHeterocycles",
    "NumAromaticCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticCarbocycles",
    "NumHeteroatoms",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "BalabanJ",
    "BertzCT",
]

PROPERTY_NORMALIZERS = {
    "MolWt": 500.0,
    "MolLogP": 10.0,
    "MolMR": 150.0,
    "TPSA": 200.0,
    "NumHAcceptors": 20.0,
    "NumHDonors": 10.0,
    "NumRotatableBonds": 20.0,
    "NumAromaticRings": 10.0,
    "NumSaturatedRings": 10.0,
    "NumAliphaticRings": 10.0,
    "NumAromaticHeterocycles": 10.0,
    "NumAromaticCarbocycles": 10.0,
    "NumSaturatedHeterocycles": 10.0,
    "NumSaturatedCarbocycles": 10.0,
    "NumAliphaticHeterocycles": 10.0,
    "NumAliphaticCarbocycles": 10.0,
    "NumHeteroatoms": 50.0,
    "NumValenceElectrons": 500.0,
    "NumRadicalElectrons": 10.0,
    "BalabanJ": 5.0,
    "BertzCT": 3000.0,
}


def mol_property_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute molecular property descriptors.

    Returns array of shape (21,) with normalized values.
    """
    props = []
    for name in PROPERTY_NAMES:
        func = getattr(Descriptors, name, None) or getattr(GraphDescriptors, name, None)
        if func is None:
            props.append(0.0)
            continue
        try:
            val = func(mol)
            if np.isnan(val) or np.isinf(val):
                val = 0.0
        except Exception:
            val = 0.0
        norm = PROPERTY_NORMALIZERS.get(name, 1.0)
        props.append(val / norm)
    return np.array(props, dtype=np.float32)


# ---------------------------------------------------------------------------
# PyG Data wrapper
# ---------------------------------------------------------------------------


class RDKitFeatureComputer:
    """Compute RDKit-based features for a batch of ZINC graphs."""

    def __init__(self):
        self.atom_context_dim = 158
        self.mol_property_dim = len(PROPERTY_NAMES)

    def compute_for_data(self, data) -> tuple:
        """Compute features for a single PyG Data object.

        Returns:
            (atom_context, mol_properties) or (None, None) if fails
        """
        mol = graph_to_rdkit_mol(
            data.x, data.edge_index, data.edge_attr, data.num_nodes
        )
        if mol is None:
            return None, None

        try:
            atom_ctx = mol_atom_context_features(mol)
            mol_props = mol_property_descriptors(mol)
            return atom_ctx, mol_props
        except Exception:
            return None, None

    def compute_fingerprints(self, data, fp_dim: int = 512) -> tuple:
        """Compute molecular and scaffold ECFP fingerprints.

        Returns:
            (mol_fp, scaff_fp) as numpy arrays of shape (fp_dim,)
            or (None, None) if fails.
        """
        mol = graph_to_rdkit_mol(
            data.x, data.edge_index, data.edge_attr, data.num_nodes
        )
        if mol is None:
            return None, None

        try:
            mol_fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=fp_dim
            )
            mol_fp_arr = np.zeros((fp_dim,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(mol_fp, mol_fp_arr)

            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold.GetNumAtoms() == 0:
                scaff_fp_arr = np.zeros((fp_dim,), dtype=np.float32)
            else:
                scaff_fp = AllChem.GetMorganFingerprintAsBitVect(
                    scaffold, radius=2, nBits=fp_dim
                )
                scaff_fp_arr = np.zeros((fp_dim,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(scaff_fp, scaff_fp_arr)

            return mol_fp_arr, scaff_fp_arr
        except Exception:
            return None, None
