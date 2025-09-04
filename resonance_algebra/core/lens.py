"""Lens: Orthonormal basis for spectral projection"""

import numpy as np
from typing import Optional


class Lens:
    """
    A lens is an orthonormal basis that projects vectors into spectral space.
    
    Think of it as a frequency analyzer that decomposes signals into bands.
    """
    
    def __init__(self, basis: np.ndarray, name: Optional[str] = None):
        """
        Initialize a Lens with an orthonormal basis.
        
        Args:
            basis: (d, r) matrix with orthonormal columns
            name: Optional name for the lens
        """
        self.B = basis
        self.name = name or "unnamed"
        self.d, self.r = basis.shape
        
        # Verify orthonormality
        if not np.allclose(self.B.T @ self.B, np.eye(self.r), atol=1e-6):
            # Orthonormalize if needed
            q, _ = np.linalg.qr(self.B)
            self.B = q[:, :self.r]
    
    @classmethod
    def random(cls, d: int, r: int, seed: Optional[int] = None, name: Optional[str] = None):
        """Create a random orthonormal lens."""
        rng = np.random.default_rng(seed)
        M = rng.normal(size=(d, r))
        Q, _ = np.linalg.qr(M)
        return cls(Q[:, :r], name)
    
    def project(self, v: np.ndarray) -> np.ndarray:
        """Project vector v into spectral space."""
        return self.B.T @ v
    
    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct vector from spectral coefficients."""
        return self.B @ coeffs
    
    def __repr__(self):
        return f"Lens(name='{self.name}', d={self.d}, r={self.r})"