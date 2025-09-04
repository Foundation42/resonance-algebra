"""Core operations of Resonance Algebra"""

import numpy as np
from typing import Optional, Tuple
from .concept import Concept
from .lens import Lens


def spectral_projection(v: np.ndarray, lens: Lens) -> np.ndarray:
    """Project a vector through a lens into spectral space."""
    return lens.project(v)


def resonance(
    x: Concept, 
    y: Concept, 
    lens: Lens,
    weights: Optional[np.ndarray] = None,
    normalize: bool = True
) -> Tuple[float, float]:
    """
    Calculate resonance between two concepts under a lens.
    
    Returns:
        (inner_product, coherence)
    """
    X = lens.project(x.v).astype(np.complex128)
    Y = lens.project(y.v).astype(np.complex128)
    
    if weights is None:
        weights = np.ones_like(X)
    
    # Hermitian inner product in band space
    inner = np.sum(weights * X * np.conj(Y))
    inner_real = float(np.real(inner))
    
    if not normalize:
        return inner_real, float('nan')
    
    # Compute coherence (normalized resonance)
    nx = float(np.sqrt(np.real(np.sum(weights * X * np.conj(X)))))
    ny = float(np.sqrt(np.real(np.sum(weights * Y * np.conj(Y)))))
    
    if nx == 0 or ny == 0:
        return inner_real, 0.0
    
    coherence = inner_real / (nx * ny)
    return inner_real, coherence


def bind_phase(x: Concept, lens: Lens, phase: np.ndarray) -> Concept:
    """
    Bind phases to spectral bands (complex multiplication).
    
    This is the core compositional operation - like adding a "role" to a concept.
    """
    X = lens.project(x.v).astype(np.complex128)
    X_bound = X * np.exp(1j * phase)
    v_new = lens.reconstruct(X_bound)
    
    return Concept(
        x.modality,
        np.real(v_new),
        metadata={'operation': 'bind', 'phase': phase}
    )


def unbind_phase(x: Concept, lens: Lens, phase: np.ndarray) -> Concept:
    """
    Unbind phases from spectral bands (inverse of bind).
    
    Removes a "role" from a concept.
    """
    return bind_phase(x, lens, -phase)


def condition(x: Concept, lens: Lens, weights: np.ndarray) -> Concept:
    """
    Condition a concept by reweighting its spectral bands.
    
    This is like applying a filter or "sieve" to emphasize certain frequencies.
    """
    X = lens.project(x.v).astype(np.complex128)
    X_conditioned = weights * X
    v_new = lens.reconstruct(X_conditioned)
    
    return Concept(
        x.modality,
        np.real(v_new),
        metadata={'operation': 'condition', 'weights': weights}
    )


def mix(x: Concept, y: Concept, alpha: float = 0.5) -> Concept:
    """
    Mix two concepts (convex combination).
    
    Args:
        alpha: Mix weight for x (1-alpha for y)
    """
    if x.modality != y.modality:
        raise ValueError("Can only mix concepts from same modality")
    
    v_mixed = alpha * x.v + (1 - alpha) * y.v
    
    return Concept(
        x.modality,
        v_mixed,
        metadata={'operation': 'mix', 'alpha': alpha}
    )