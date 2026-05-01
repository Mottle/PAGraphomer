"""Scaffold-invariant molecular perturbation using RDKit (CReM alternative).

Implements fragment-based molecular perturbation that preserves the Murcko scaffold.
Uses BRICS decomposition and a simple fragment replacement strategy.
Also computes ECFP fingerprints for contrastive learning.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


# Simple fragment library for replacement (commonly occurring fragments)
SIMPLE_FRAGMENTS = [
    "[*:1]C",           # methyl
    "[*:1]CC",          # ethyl
    "[*:1]O",           # hydroxyl
    "[*:1]N",           # amino
    "[*:1]F",           # fluoro
    "[*:1]Cl",          # chloro
    "[*:1]Br",          # bromo
    "[*:1]C(=O)O",      # carboxyl
    "[*:1]C(=O)N",      # amide
    "[*:1]S(=O)(=O)N",  # sulfonamide
    "[*:1]CN",          # aminomethyl
    "[*:1]OC",          # methoxy
    "[*:1]C(F)(F)F",    # trifluoromethyl
    "[*:1]C#N",         # cyano
    "[*:1]N(C)C",       # dimethylamino
]


def compute_ecfp(mol: Chem.Mol, radius: int = 2, nBits: int = 512) -> np.ndarray:
    """Compute ECFP fingerprint as numpy array.
    
    Args:
        mol: RDKit Mol object
        radius: Morgan fingerprint radius
        nBits: number of bits
        
    Returns:
        Binary fingerprint as float32 numpy array [nBits]
    """
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_scaffold_ecfp(mol: Chem.Mol, radius: int = 2, nBits: int = 512) -> np.ndarray:
    """Compute ECFP fingerprint of the Murcko scaffold."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return np.zeros(nBits, dtype=np.float32)
    return compute_ecfp(scaffold, radius=radius, nBits=nBits)


def get_scaffold_smiles(mol: Chem.Mol) -> str:
    """Get Murcko scaffold SMILES."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold)


def get_scaffold_atom_indices(mol: Chem.Mol) -> set:
    """Get atom indices belonging to the Murcko scaffold."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return set()
    
    # Find scaffold substructure match in original molecule
    match = mol.GetSubstructMatch(scaffold)
    return set(match)


def get_scaffold_atom_indices(mol: Chem.Mol) -> set:
    """Get atom indices belonging to the Murcko scaffold."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return set()
    
    # Find scaffold substructure match in original molecule
    match = mol.GetSubstructMatch(scaffold)
    return set(match)


def generate_perturbations(
    mol: Chem.Mol,
    num_candidates: int = 6,
    seed: Optional[int] = None,
) -> List[Chem.Mol]:
    """Generate scaffold-invariant molecular perturbations.

    Args:
        mol: RDKit Mol object
        num_candidates: number of perturbations to generate
        seed: random seed

    Returns:
        List of perturbed Mol objects (may contain fewer than num_candidates)
    """
    if seed is not None:
        random.seed(seed)
    
    scaffold_atoms = get_scaffold_atom_indices(mol)
    if not scaffold_atoms:
        return []
    
    # Use BRICS to find breakable bonds
    mol_with_dummy = BRICS.BreakBRICSBonds(mol)
    fragments = Chem.GetMolFrags(mol_with_dummy, asMols=True, sanitizeFrags=False)
    
    perturbations = []
    
    for frag in fragments:
        if frag.GetNumAtoms() < 2:
            continue
            
        # Check if this fragment is a side chain (not part of scaffold)
        frag_atoms = set()
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() > 0:  # Skip dummy atoms
                # Map back to original molecule - approximate
                pass
        
        # Simpler approach: find breakable bonds that connect scaffold to side chain
        breakable_bonds = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            in_scaff_1 = a1 in scaffold_atoms
            in_scaff_2 = a2 in scaffold_atoms
            
            # Side chain bond: one end in scaffold, one end not
            if in_scaff_1 != in_scaff_2:
                breakable_bonds.append((a1, a2))
    
    if not breakable_bonds:
        return []
    
    # Generate perturbations by replacing side chains
    attempts = 0
    max_attempts = num_candidates * 3
    
    while len(perturbations) < num_candidates and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a breakable bond
        a1, a2 = random.choice(breakable_bonds)
        
        # Determine which atom is the attachment point on scaffold
        if a1 in scaffold_atoms:
            scaffold_atom = a1
            side_chain_atom = a2
        else:
            scaffold_atom = a2
            side_chain_atom = a1
        
        # Create editable molecule
        emol = Chem.EditableMol(mol)
        
        # Remove the bond
        bond_idx = mol.GetBondBetweenAtoms(a1, a2).GetIdx()
        emol.RemoveBond(a1, a2)
        
        # Remove side chain atoms (BFS from side_chain_atom)
        atoms_to_remove = set()
        queue = [side_chain_atom]
        visited = {scaffold_atom}
        
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            atoms_to_remove.add(curr)
            
            atom = mol.GetAtomWithIdx(curr)
            for neighbor in atom.GetNeighbors():
                nidx = neighbor.GetIdx()
                if nidx not in scaffold_atoms and nidx not in visited:
                    queue.append(nidx)
        
        # Remove atoms in reverse order to maintain indices
        for idx in sorted(atoms_to_remove, reverse=True):
            emol.RemoveAtom(idx)
        
        # Add replacement fragment
        replacement = random.choice(SIMPLE_FRAGMENTS)
        replacement_mol = Chem.MolFromSmiles(replacement)
        
        if replacement_mol is None:
            continue
        
        # Find dummy atom in replacement
        dummy_idx = None
        for atom in replacement_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx = atom.GetIdx()
                break
        
        if dummy_idx is None:
            continue
        
        # Combine scaffold and replacement
        # Adjust scaffold_atom index after removals
        scaffold_atom_adj = scaffold_atom
        for removed in sorted(atoms_to_remove):
            if removed < scaffold_atom:
                scaffold_atom_adj -= 1
        
        try:
            combined = emol.GetMol()
            combined = Chem.RWMol(combined)
            
            # Add replacement atoms
            atom_map = {}
            for atom in replacement_mol.GetAtoms():
                if atom.GetIdx() == dummy_idx:
                    continue
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                new_idx = combined.AddAtom(new_atom)
                atom_map[atom.GetIdx()] = new_idx
            
            # Add replacement bonds
            for bond in replacement_mol.GetBonds():
                b = bond.GetBeginAtomIdx()
                e = bond.GetEndAtomIdx()
                if b == dummy_idx:
                    combined.AddBond(scaffold_atom_adj, atom_map[e], bond.GetBondType())
                elif e == dummy_idx:
                    combined.AddBond(atom_map[b], scaffold_atom_adj, bond.GetBondType())
                else:
                    combined.AddBond(atom_map[b], atom_map[e], bond.GetBondType())
            
            # Sanitize
            Chem.SanitizeMol(combined)
            
            # Verify scaffold preservation
            new_scaffold = MurckoScaffold.GetScaffoldForMol(combined)
            old_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            
            if (new_scaffold.GetNumAtoms() > 0 and 
                old_scaffold.GetNumAtoms() > 0 and
                Chem.MolToSmiles(new_scaffold) == Chem.MolToSmiles(old_scaffold)):
                perturbations.append(combined)
                
        except Exception:
            continue
    
    return perturbations


def mol_to_graph_data(mol: Chem.Mol, reference_data):
    """Convert RDKit Mol to PyG Data matching ZINC format.
    
    Args:
        mol: RDKit Mol object
        reference_data: Original PyG Data for format reference
        
    Returns:
        New PyG Data object or None if conversion fails
    """
    from torch_geometric.data import Data
    import torch
    
    try:
        # Get atom types (using same encoding as ZINC)
        atom_types = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            charge = atom.GetFormalCharge()
            # Simple mapping (this is approximate)
            atom_type = _symbol_to_zinc_type(symbol, charge)
            atom_types.append(atom_type)
        
        x = torch.tensor(atom_types, dtype=torch.long).view(-1, 1)
        
        # Get bonds
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
            
            # Bond type encoding: 1=single, 2=double, 3=triple
            bt = bond.GetBondType()
            if bt == Chem.BondType.SINGLE:
                btype = 1
            elif bt == Chem.BondType.DOUBLE:
                btype = 2
            elif bt == Chem.BondType.TRIPLE:
                btype = 3
            else:
                btype = 1
            edge_attr.extend([btype, btype])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
        
    except Exception:
        return None


def _symbol_to_zinc_type(symbol: str, charge: int) -> int:
    """Map atom symbol to ZINC atom type (approximate)."""
    # This is a simplified mapping
    symbol_charge_map = {
        ('C', 0): 0, ('O', 0): 1, ('N', 0): 2, ('F', 0): 3,
        ('C', 1): 4, ('S', 0): 5, ('Cl', 0): 6, ('O', -1): 7,
        ('N', 1): 8, ('Br', 0): 9, ('N', 3): 10, ('N', 2): 11,
        ('N', 1): 12, ('N', -1): 13, ('S', -1): 14, ('I', 0): 15,
        ('P', 0): 16, ('O', 1): 17, ('N', 0): 18, ('O', 1): 19,
        ('S', 1): 20, ('P', 1): 21, ('P', 2): 22, ('C', -1): 23,
        ('P', 1): 24, ('S', 1): 25, ('C', -1): 26, ('P', 1): 27,
    }
    return symbol_charge_map.get((symbol, charge), 0)