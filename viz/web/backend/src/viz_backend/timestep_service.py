"""
Service for loading and processing timestep data from noise and denoise processes.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("uvicorn")

class TimestepService:
    """Service for loading timestep data from noise_process and denoise_process folders"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.noise_process_dir = data_dir / "noise_process" / "raw"
        self.denoise_process_dir = data_dir / "denoise_process" / "raw"
        
        # DiGress MOSES atom_decoder (8 node classes in the npz files)
        self.atom_types = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']
        
        # Bond type mapping
        self.bond_types = {
            0: None,      # No bond
            1: 'SINGLE',
            2: 'DOUBLE', 
            3: 'TRIPLE',
            4: 'AROMATIC'
        }
        
        logger.info(f"TimestepService initialized")
        logger.info(f"Noise process dir: {self.noise_process_dir}")
        logger.info(f"Denoise process dir: {self.denoise_process_dir}")
        
    def get_process_dir(self, process_type: str) -> Path:
        """Get directory for specified process type"""
        if process_type == "noise":
            return self.noise_process_dir
        elif process_type == "denoise":
            return self.denoise_process_dir
        else:
            raise ValueError(f"Unknown process type: {process_type}. Must be 'noise' or 'denoise'")
    
    def load_timestep_npz(self, process_type: str, timestep: int) -> Dict:
        """
        Load NPZ file for given process and timestep.
        
        Args:
            process_type: 'noise' or 'denoise'
            timestep: Timestep number (0-500)
            
        Returns:
            Dictionary with 'nodes' and 'edges' arrays
        """
        process_dir = self.get_process_dir(process_type)
        
        # Determine filename based on timestep
        if timestep == 0:
            if process_type == "noise":
                filename = "step_00_original.npz"
            else:  # denoise
                filename = "step_00_fully_noisy.npz"
        else:
            # Use 2-digit format for timesteps 1-99, 3-digit for 100-500
            if timestep < 100:
                timestep_str = f"{timestep:02d}"
            else:
                timestep_str = str(timestep)
            
            if process_type == "noise":
                filename = f"step_{timestep_str}_noisy.npz"
            else:  # denoise
                filename = f"step_{timestep_str}_denoised.npz"
        
        filepath = process_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Timestep file not found: {filepath}")
        
        # Load NPZ file
        data = np.load(filepath)
        nodes = data['nodes'] if 'nodes' in data else data['X']  # Support both formats
        edges = data['edges'] if 'edges' in data else data['E']

        # Forward-process files for t >= 1 store class probabilities q(x_t | x_0)
        # (class axis last), not indices. Turn them into one index graph.
        if nodes.ndim == 2:
            nodes, edges = self._sample_from_probs(nodes, edges)
        elif process_type == "noise" and timestep == 0:
            # step_00_original stores padding nodes as class 0 (C). Take the node
            # mask from step 1, so padding does not render as extra atoms.
            mask_file = process_dir / "step_01_noisy.npz"
            if mask_file.exists():
                mask = np.load(mask_file)['nodes'].sum(-1) > 0.5
                nodes = np.where(mask, nodes, -1)

        return {'nodes': nodes, 'edges': edges}

    @staticmethod
    def _sample_from_probs(node_probs: np.ndarray, edge_probs: np.ndarray, seed: int = 0):
        """Sample one graph from per-node and per-edge class probabilities.

        The same uniforms (fixed seed) are used for every timestep, with inverse-CDF
        sampling. So each frame is a valid sample of q(x_t | x_0), and consecutive
        frames differ only where the probabilities moved, which makes the noise
        view change gradually. Padding nodes (all-zero rows) become -1.
        (Simpler fallback: nodes = node_probs.argmax(-1), edges = edge_probs.argmax(-1).)
        """
        n = node_probs.shape[0]
        rng = np.random.default_rng(seed)
        u_nodes = rng.random(n)
        u_edges = rng.random((n, n))

        node_mask = node_probs.sum(-1) > 0.5
        node_cdf = np.cumsum(node_probs, axis=-1)
        nodes = (u_nodes[:, None] * node_cdf[:, -1:] > node_cdf).sum(-1)
        nodes = np.minimum(nodes, node_probs.shape[-1] - 1)
        nodes[~node_mask] = -1

        edge_cdf = np.cumsum(edge_probs, axis=-1)
        edges = (u_edges[..., None] * edge_cdf[..., -1:] > edge_cdf).sum(-1)
        edges = np.minimum(edges, edge_probs.shape[-1] - 1)
        edges[edge_probs.sum(-1) < 0.5] = 0  # diagonal and padding: no bond
        edges = np.triu(edges, 1)
        edges = edges + edges.T  # symmetric, from the upper triangle
        return nodes, edges
    
    def convert_npz_to_molecule_format(self, npz_data: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Convert NPZ data to MoleculeVisualizer format.
        
        Args:
            npz_data: Dictionary with 'nodes' and 'edges' arrays
            
        Returns:
            Tuple of (atoms_data, bonds_data) for MoleculeVisualizer
        """
        nodes = npz_data['nodes']
        edges = npz_data['edges']
        
        # Convert nodes to atoms
        # nodes is 1D array of atom type indices, -1 means padding (no atom)
        atoms_data = []
        for idx, atom_type_idx in enumerate(nodes):
            # Skip padding atoms
            if atom_type_idx < 0:
                continue
                
            atom_type_idx = int(atom_type_idx)
            
            # Get atom symbol
            if atom_type_idx < len(self.atom_types):
                symbol = self.atom_types[atom_type_idx]
            else:
                symbol = 'C'  # Default to Carbon
            
            atoms_data.append({
                'index': len(atoms_data),  # Use sequential index (not original idx with padding)
                'original_index': idx,  # Keep track of original index for edges
                'symbol': symbol,
                'element': symbol,
                'atomic_num': self._get_atomic_number(symbol)
            })
        
        # Create mapping from original index to new index (without padding)
        original_to_new = {}
        for new_idx, atom in enumerate(atoms_data):
            original_to_new[atom['original_index']] = new_idx
        
        # Convert edges to bonds
        # edges is 2D adjacency matrix [num_atoms, num_atoms] with bond type values
        bonds_data = []
        num_nodes = len(nodes)
        
        for i in range(num_nodes):
            # Skip if this is a padding atom
            if nodes[i] < 0:
                continue
                
            for j in range(i + 1, num_nodes):  # Only upper triangle to avoid duplicates
                # Skip if this is a padding atom
                if nodes[j] < 0:
                    continue
                    
                # Get bond type from adjacency matrix
                bond_type_idx = int(edges[i, j])
                
                # Skip if no bond (bond_type = 0)
                if bond_type_idx == 0:
                    continue
                
                bond_type = self.bond_types.get(bond_type_idx, 'SINGLE')
                
                bonds_data.append({
                    'begin_atom': original_to_new[i],
                    'end_atom': original_to_new[j],
                    'bond_type': bond_type_idx,
                    'bond_type_name': bond_type
                })
        
        return atoms_data, bonds_data
    
    def _get_atomic_number(self, symbol: str) -> int:
        """Get atomic number for element symbol"""
        atomic_numbers = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16,
            'Cl': 17, 'Br': 35
        }
        return atomic_numbers.get(symbol, 6)
    
    def get_molecule_at_timestep(self, process_type: str, timestep: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Get molecule data at specific timestep in MoleculeVisualizer format.
        
        Args:
            process_type: 'noise' or 'denoise'
            timestep: Timestep number (0-500)
            
        Returns:
            Tuple of (atoms_data, bonds_data)
        """
        npz_data = self.load_timestep_npz(process_type, timestep)
        return self.convert_npz_to_molecule_format(npz_data)
    
    def get_available_timesteps(self, process_type: str) -> List[int]:
        """Get list of available timesteps for process"""
        process_dir = self.get_process_dir(process_type)
        
        if not process_dir.exists():
            return []
        
        timesteps = []
        for filepath in process_dir.glob("*.npz"):
            # Extract timestep number from filename
            stem = filepath.stem
            if "step_" in stem:
                try:
                    parts = stem.split("_")
                    timestep = int(parts[1])
                    timesteps.append(timestep)
                except (ValueError, IndexError):
                    continue
        
        return sorted(timesteps)
    
    def get_statistics(self) -> Dict:
        """Get statistics about available data"""
        noise_timesteps = self.get_available_timesteps("noise")
        denoise_timesteps = self.get_available_timesteps("denoise")
        
        return {
            "noise_process": {
                "available": len(noise_timesteps) > 0,
                "num_timesteps": len(noise_timesteps),
                "timestep_range": [min(noise_timesteps), max(noise_timesteps)] if noise_timesteps else None,
                "direction": "Clean (t=0) → Noisy (t=500)"
            },
            "denoise_process": {
                "available": len(denoise_timesteps) > 0,
                "num_timesteps": len(denoise_timesteps),
                "timestep_range": [min(denoise_timesteps), max(denoise_timesteps)] if denoise_timesteps else None,
                "direction": "Noisy (t=0) → Clean (t=500)"
            }
        }
