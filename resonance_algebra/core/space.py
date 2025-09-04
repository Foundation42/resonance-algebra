"""ResonanceSpace: Container for modalities and their lenses"""

import numpy as np
from typing import Dict, Optional
from .lens import Lens
from .concept import Concept


class ResonanceSpace:
    """
    A space containing multiple modalities, lenses, and alignment maps.
    
    This is the main workspace for resonance algebra operations.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.modalities: Dict[str, dict] = {}  # modality -> {d, lenses}
        self.alignments: Dict[tuple, np.ndarray] = {}  # (src, dst) -> alignment matrix
        
    def add_modality(self, name: str, dimension: int):
        """Add a modality space."""
        self.modalities[name] = {
            'd': dimension,
            'lenses': {}
        }
        
    def add_lens(self, modality: str, lens: Lens):
        """Add a lens to a modality."""
        if modality not in self.modalities:
            raise ValueError(f"Modality '{modality}' not found")
        self.modalities[modality]['lenses'][lens.name] = lens
        
    def get_lens(self, modality: str, lens_name: str) -> Lens:
        """Retrieve a lens."""
        return self.modalities[modality]['lenses'][lens_name]
    
    def add_alignment(self, src_modality: str, dst_modality: str, matrix: np.ndarray):
        """Add an alignment map between modalities."""
        self.alignments[(src_modality, dst_modality)] = matrix
        
    def align(self, concept: Concept, target_modality: str) -> Concept:
        """Align a concept to a different modality."""
        key = (concept.modality, target_modality)
        if key not in self.alignments:
            raise ValueError(f"No alignment from {concept.modality} to {target_modality}")
        
        aligned_v = concept.v @ self.alignments[key]
        return Concept(target_modality, aligned_v)
    
    def project(self, concept: Concept, lens_name: str) -> np.ndarray:
        """Project a concept through a lens."""
        lens = self.get_lens(concept.modality, lens_name)
        coeffs = lens.project(concept.v)
        concept.add_spectrum(lens_name, coeffs)
        return coeffs
    
    def __repr__(self):
        mods = list(self.modalities.keys())
        return f"ResonanceSpace(name='{self.name}', modalities={mods})"