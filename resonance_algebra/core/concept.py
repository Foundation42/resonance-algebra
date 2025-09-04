"""Concept: A vector in modality space with spectral decomposition"""

import numpy as np
from typing import Dict, Optional


class Concept:
    """
    A concept is an embedding that can be viewed through different lenses.
    
    It carries both a vector representation and optional spectral decompositions.
    """
    
    def __init__(
        self, 
        modality: str,
        vector: np.ndarray,
        spectra: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[dict] = None
    ):
        """
        Initialize a Concept.
        
        Args:
            modality: The modality (text, image, audio, etc.)
            vector: The embedding vector
            spectra: Optional pre-computed spectral decompositions
            metadata: Optional metadata about the concept
        """
        self.modality = modality
        self.v = vector
        self.spectra = spectra or {}
        self.metadata = metadata or {}
        self.d = len(vector)
        
    def add_spectrum(self, lens_name: str, coeffs: np.ndarray):
        """Cache a spectral decomposition under a lens."""
        self.spectra[lens_name] = coeffs
        
    def get_spectrum(self, lens_name: str) -> Optional[np.ndarray]:
        """Retrieve cached spectral decomposition."""
        return self.spectra.get(lens_name)
    
    def clone(self):
        """Create a deep copy of the concept."""
        return Concept(
            self.modality,
            self.v.copy(),
            {k: v.copy() for k, v in self.spectra.items()},
            self.metadata.copy()
        )
    
    def __repr__(self):
        spectra_str = f", spectra={list(self.spectra.keys())}" if self.spectra else ""
        return f"Concept(modality='{self.modality}', d={self.d}{spectra_str})"