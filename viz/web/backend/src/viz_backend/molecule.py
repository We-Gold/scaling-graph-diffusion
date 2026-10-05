"""
Diffusion data processing module for converting NPZ files to graph data.

This module provides testable components for processing diffusion model output
into graph data suitable for visualization and API consumption.
"""

from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from typing import List, Optional, Tuple, Dict, Union
import json
import logging

logger = logging.getLogger(__name__)


class AtomDecoder:
    """Handles loading and managing atom type mappings."""
    
    def __init__(self, decoder_file: Path):
        self.decoder_file = decoder_file
        self._atom_type_map = None
    
    def load(self) -> Optional[Dict[int, str]]:
        """
        Load atom decoder from the npz file.
        
        Returns:
            Dictionary mapping atom type indices to atomic symbols
        """
        if self._atom_type_map is not None:
            return self._atom_type_map
            
        try:
            decoder_data = np.load(self.decoder_file, allow_pickle=True)
            logger.info(f"Keys in atom_decoder.npz: {decoder_data.files}")
            
            # Log decoder information
            for key in decoder_data.files:
                logger.info(f"Decoder key '{key}' shape: {decoder_data[key].shape}")
                logger.debug(f"Decoder key '{key}' content: {decoder_data[key]}")
            
            # Try common key names for atom decoder
            possible_keys = ['atom_decoder', 'atoms', 'atom_types', 'decoder']
            atom_decoder = None
            
            for key in possible_keys:
                if key in decoder_data.files:
                    atom_decoder = decoder_data[key]
                    break
            
            if atom_decoder is None:
                # If no standard key found, use the first available key
                if len(decoder_data.files) > 0:
                    key = decoder_data.files[0]
                    atom_decoder = decoder_data[key]
                    logger.info(f"Using key '{key}' as atom decoder")
            
            if atom_decoder is not None:
                # Create mapping dictionary
                atom_type_map = {}
                for i, atom_symbol in enumerate(atom_decoder):
                    if isinstance(atom_symbol, bytes):
                        atom_symbol = atom_symbol.decode('utf-8')
                    elif isinstance(atom_symbol, np.str_) or isinstance(atom_symbol, str):
                        atom_symbol = str(atom_symbol)
                    atom_type_map[i] = atom_symbol
                    logger.debug(f"Atom type {i}: {atom_symbol}")
                
                self._atom_type_map = atom_type_map
                return atom_type_map
            else:
                logger.error("Could not find atom decoder data")
                return None
                
        except Exception as e:
            logger.error(f"Error loading atom decoder: {e}")
            return None
    
    @property
    def atom_type_map(self) -> Optional[Dict[int, str]]:
        """Get the cached atom type map."""
        return self._atom_type_map


class NPZFileLoader:
    """Handles loading NPZ files from directories."""
    
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
    
    def load_all_files(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """
        Load all NPZ files from the raw directory.
        
        Returns:
            Tuple of (nodes_list, edges_list, filenames)
        """
        npz_files = sorted(self.raw_dir.glob("*.npz"))
        logger.info(f"Found {len(npz_files)} NPZ files in {self.raw_dir}")
        
        nodes_list = []
        edges_list = []
        filenames = []
        
        for npz_file in npz_files:
            try:
                nodes, edges = self._load_single_file(npz_file)
                if nodes is not None and edges is not None:
                    nodes_list.append(nodes)
                    edges_list.append(edges)
                    filenames.append(npz_file.name)
                    
                    if len(nodes_list) % 50 == 0:
                        logger.info(f"Loaded {len(nodes_list)} files...")
                        
            except Exception as e:
                logger.warning(f"Error loading {npz_file.name}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(nodes_list)} NPZ files")
        return nodes_list, edges_list, filenames
    
    def _load_single_file(self, npz_file: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a single NPZ file and extract nodes and edges."""
        npz_data = np.load(npz_file, allow_pickle=True)
        
        # Get nodes and edges data
        if 'nodes' not in npz_data.files or 'edges' not in npz_data.files:
            logger.warning(f"{npz_file.name} missing 'nodes' or 'edges' keys")
            return None, None
        
        nodes = npz_data['nodes']
        edges = npz_data['edges']
        
        # Handle different data structures - if 3D, take first sample
        if len(nodes.shape) > 1:
            nodes = nodes[0] if nodes.shape[0] > 0 else nodes.flatten()
        if len(edges.shape) > 2:
            edges = edges[0] if edges.shape[0] > 0 else edges.reshape(edges.shape[-2:])
        
        return nodes, edges


class MoleculeGraphConverter:
    """Converts NPZ data to graph representations suitable for frontend."""
    
    @staticmethod
    def get_atomic_number(symbol: str) -> int:
        """Get atomic number from element symbol."""
        atomic_numbers = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
            'S': 16, 'Cl': 17, 'Br': 35, 'I': 53
        }
        return atomic_numbers.get(symbol, 6)  # Default to Carbon if unknown

    @staticmethod
    def get_bond_order(bond_type: int) -> float:
        """Convert bond type to bond order."""
        bond_orders = {
            1: 1.0,  # Single
            2: 2.0,  # Double
            3: 3.0,  # Triple
            4: 1.5   # Aromatic (approximate)
        }
        return bond_orders.get(bond_type, 1.0)
    
    def process_single_sample(self, sample_nodes: np.ndarray, sample_edges: np.ndarray, 
                            atom_type_map: Dict[int, str], sample_id: int) -> Dict:
        """
        Process a single sample and convert to graph data.
        
        Args:
            sample_nodes: Node array for the sample
            sample_edges: Edge array for the sample
            atom_type_map: Mapping from atom type indices to symbols
            sample_id: Unique identifier for this sample
        
        Returns:
            Dictionary with graph data
        """
        # Check if sample has any atoms
        num_valid_atoms = np.sum(sample_nodes != -1)
        if num_valid_atoms == 0:
            return {
                'sample_id': sample_id,
                'atoms': [],
                'bonds': [],
                'num_atoms': 0,
                'num_bonds': 0,
                'is_valid': False,
                'error_type': 'no_atoms'
            }
        
        # Extract atoms data
        atoms, atom_index_map = self._extract_atoms(sample_nodes, atom_type_map)
        
        # Extract bonds data
        bonds = self._extract_bonds(sample_edges, atom_index_map)
        
        # Determine if structure is valid
        is_valid = len(atoms) > 0
        error_type = "success" if is_valid else "invalid_structure"
        
        return {
            'sample_id': sample_id,
            'atoms': atoms,
            'bonds': bonds,
            'num_atoms': len(atoms),
            'num_bonds': len(bonds),
            'is_valid': is_valid,
            'error_type': error_type
        }
    
    def _extract_atoms(self, sample_nodes: np.ndarray, atom_type_map: Dict[int, str]) -> Tuple[List[Dict], Dict[int, int]]:
        """Extract atoms data from node array."""
        atoms = []
        atom_index_map = {}  # Map from original index to new index
        new_index = 0
        
        for orig_idx, atom_type in enumerate(sample_nodes):
            if atom_type != -1 and atom_type in atom_type_map:
                atoms.append({
                    'id': new_index,
                    'symbol': atom_type_map[atom_type],
                    'atomic_num': self.get_atomic_number(atom_type_map[atom_type]),
                    'original_idx': orig_idx
                })
                atom_index_map[orig_idx] = new_index
                new_index += 1
        
        return atoms, atom_index_map
    
    def _extract_bonds(self, sample_edges: np.ndarray, atom_index_map: Dict[int, int]) -> List[Dict]:
        """Extract bonds data from edge array."""
        bonds = []
        bond_id = 0
        
        for i_node in range(len(sample_edges)):
            for j_node in range(i_node + 1, len(sample_edges[i_node])):  # Upper triangle only
                bond_type = sample_edges[i_node][j_node]
                
                # Only process valid bonds (> 0)
                if bond_type > 0 and i_node in atom_index_map and j_node in atom_index_map:
                    bonds.append({
                        'id': bond_id,
                        'begin_atom': atom_index_map[i_node],
                        'end_atom': atom_index_map[j_node],
                        'bond_type': int(bond_type),
                        'bond_order': self.get_bond_order(bond_type)
                    })
                    bond_id += 1
        
        return bonds


class DiffusionDataProcessor:
    """Main processor class that orchestrates the conversion pipeline."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.atom_decoder = AtomDecoder(data_dir / "atom_decoder.npz")
        self.converter = MoleculeGraphConverter()
    
    def process_diffusion_samples(self, sample_dir: Path, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Process all diffusion samples and convert to graph data DataFrame.
        
        Args:
            sample_dir: Directory containing NPZ files
            max_samples: Maximum number of samples to process (None for all)
        
        Returns:
            DataFrame with columns: sample_id, atoms, bonds, num_atoms, num_bonds, is_valid, error_type
        """
        # Load atom decoder
        atom_type_map = self.atom_decoder.load()
        if atom_type_map is None:
            raise ValueError("Failed to load atom decoder")
        
        # Load NPZ files
        loader = NPZFileLoader(sample_dir / "denoise_process" / "raw")
        nodes_list, edges_list, filenames = loader.load_all_files()
        
        if not nodes_list:
            raise ValueError("No valid NPZ files found")
        
        # Process samples
        num_samples = min(len(nodes_list), max_samples or len(nodes_list))
        logger.info(f"Converting {num_samples} samples to graph data...")
        
        results = []
        error_counts = {"success": 0, "no_atoms": 0, "invalid_structure": 0}
        
        for i in range(num_samples):
            if i % 50 == 0:
                logger.info(f"Processing sample {i}/{num_samples}")
            
            result = self.converter.process_single_sample(
                nodes_list[i], edges_list[i], atom_type_map, i
            )
            results.append(result)
            error_counts[result['error_type']] += 1
        
        df = pd.DataFrame(results)
        
        # Log statistics
        self._log_processing_stats(num_samples, error_counts, df)
        
        return df
    
    def _log_processing_stats(self, num_samples: int, error_counts: Dict[str, int], df: pd.DataFrame):
        """Log detailed processing statistics."""
        logger.info("=== Processing Results ===")
        logger.info(f"Total samples: {num_samples}")
        for error_type, count in error_counts.items():
            percentage = (count / num_samples) * 100
            logger.info(f"{error_type}: {count} ({percentage:.1f}%)")
        
        # Additional statistics for valid molecules
        valid_df = df[df['is_valid']]
        if len(valid_df) > 0:
            logger.info("=== Valid Molecules Statistics ===")
            logger.info(f"Average atoms: {valid_df['num_atoms'].mean():.1f}")
            logger.info(f"Average bonds: {valid_df['num_bonds'].mean():.1f}")
            logger.info(f"Atom range: {valid_df['num_atoms'].min()} - {valid_df['num_atoms'].max()}")
            logger.info(f"Bond range: {valid_df['num_bonds'].min()} - {valid_df['num_bonds'].max()}")
    
    def save_to_parquet(self, df: pd.DataFrame, filepath: Path):
        """Save DataFrame to Parquet format using PyArrow."""
        # Convert complex nested data to JSON strings for Parquet storage
        df_copy = df.copy()
        df_copy['atoms'] = df_copy['atoms'].apply(lambda x: json.dumps(x) if x else "[]")
        df_copy['bonds'] = df_copy['bonds'].apply(lambda x: json.dumps(x) if x else "[]")
        
        table = pa.Table.from_pandas(df_copy)
        pq.write_table(table, filepath)
        logger.info(f"Saved {len(df_copy)} records to {filepath}")


class MoleculeVisualizer:
    """Handles molecule visualization using RDKit."""
    
    @staticmethod
    def create_molecule_from_graph_data(atoms_data: List[Dict], bonds_data: List[Dict]):
        """
        Create RDKit molecule from graph data for visualization.
        
        Args:
            atoms_data: List of atom dictionaries
            bonds_data: List of bond dictionaries
        
        Returns:
            RDKit Mol object or None
        """
        try:
            # Create editable molecule
            mol = Chem.EditableMol(Chem.Mol())
            
            # Bond type mapping
            bond_type_map = {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
                4: Chem.BondType.AROMATIC,
            }
            
            # Add atoms
            for atom_data in atoms_data:
                atom = Chem.Atom(atom_data['symbol'])
                mol.AddAtom(atom)
            
            # Add bonds
            for bond_data in bonds_data:
                bond_type = bond_data['bond_type']
                if bond_type in bond_type_map:
                    mol.AddBond(
                        bond_data['begin_atom'], 
                        bond_data['end_atom'], 
                        bond_type_map[bond_type]
                    )
            
            # Convert to molecule (skip sanitization for noisy data)
            mol = mol.GetMol()
            return mol
        
        except Exception as e:
            logger.error(f"Error creating molecule from graph data: {e}")
            return None
    
    @classmethod
    def visualize_molecule_from_graph_data(cls, atoms_data: List[Dict], bonds_data: List[Dict], 
                                         title: str = "Molecule", save_path: Optional[Path] = None):
        """Visualize molecule directly from graph data."""
        mol = cls.create_molecule_from_graph_data(atoms_data, bonds_data)
        
        if mol is None:
            logger.warning("Cannot visualize: failed to create molecule from graph data")
            return
        
        try:
            # Generate 2D coordinates
            Chem.rdDepictor.Compute2DCoords(mol)
            
            # Create image
            img = Draw.MolToImage(mol, size=(400, 400))
            
            # Display using matplotlib
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(title)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            plt.show()
            logger.info("✅ Successfully visualized molecule from graph data")
            
        except Exception as e:
            logger.error(f"Error visualizing molecule: {e}")