"""
Resonance Algebra - A spectral approach to embeddings and neural computation

A breakthrough framework treating embeddings as spectral compositions rather than flat vectors,
enabling gradient-free computation through phase geometry.
"""

__version__ = "0.1.0"
__author__ = "Christian Beneš & Claude"

from .core import *
from .gates import *

__all__ = ['ResonanceSpace', 'Lens', 'Concept', 'PhaseLogic']