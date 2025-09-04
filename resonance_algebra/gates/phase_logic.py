"""Phase-based logic gates using Resonance Algebra"""

import numpy as np
from typing import Optional, Tuple
from ..core import Lens


class PhaseGate:
    """Base class for phase-based logic gates."""
    
    def __init__(self, name: str):
        self.name = name
    
    def __call__(self, *inputs):
        raise NotImplementedError


class PhaseLogic:
    """
    Implements Boolean logic through phase algebra.
    
    Key insight: Boolean operations emerge naturally from phase interference
    and spectral decomposition, requiring no learning or backpropagation.
    """
    
    def __init__(self, d: int = 32, r: int = 8, seed: Optional[int] = None):
        """
        Initialize phase-based logic system.
        
        Args:
            d: Embedding dimension
            r: Number of spectral bands
            seed: Random seed for reproducibility
        """
        self.d = d
        self.r = r
        self.lens = Lens.random(d, r, seed=seed, name="logic")
        
        # Band assignments
        self.band_a = 0      # First input
        self.band_b = 1      # Second input
        self.band_diff = 2   # Interference (XOR)
        
    @staticmethod
    def bit_to_phase(bit: int, noise: float = 0.0) -> complex:
        """
        Encode bit as phase: 0 → 0°, 1 → 180°
        
        Args:
            bit: Binary value (0 or 1)
            noise: Phase noise standard deviation (radians)
        """
        base_phase = 0.0 if bit == 0 else np.pi
        if noise > 0:
            phase = base_phase + noise * np.random.randn()
        else:
            phase = base_phase
        return np.exp(1j * phase)
    
    @staticmethod
    def phase_to_bit(z: complex, threshold: float = 0.0) -> int:
        """Decode phase to bit via real part sign."""
        return 0 if np.real(z) >= threshold else 1
    
    def encode_inputs(self, a: int, b: int, noise: float = 0.0) -> np.ndarray:
        """Encode two bits into spectral space."""
        coeffs = np.zeros(self.r, dtype=complex)
        
        # Individual phases
        za = self.bit_to_phase(a, noise)
        zb = self.bit_to_phase(b, noise)
        
        # Encode in bands
        coeffs[self.band_a] = za
        coeffs[self.band_b] = zb
        coeffs[self.band_diff] = za * np.conj(zb)  # Phase difference
        
        return self.lens.reconstruct(coeffs)
    
    def read_bands(self, v: np.ndarray) -> Tuple[complex, complex, complex]:
        """Read phase values from spectral bands."""
        s = self.lens.project(v)
        return s[self.band_a], s[self.band_b], s[self.band_diff]
    
    # Logic gates
    
    def XOR(self, a: int, b: int, noise: float = 0.0) -> int:
        """XOR gate via phase interference."""
        v = self.encode_inputs(a, b, noise)
        _, _, z_diff = self.read_bands(v)
        # XOR is true when phase difference is ~π
        return self.phase_to_bit(z_diff)
    
    def AND(self, a: int, b: int, noise: float = 0.0) -> int:
        """AND gate via phase product."""
        v = self.encode_inputs(a, b, noise)
        za, zb, _ = self.read_bands(v)
        # AND requires both phases at π
        s1 = 1 if np.real(za) >= 0 else -1
        s2 = 1 if np.real(zb) >= 0 else -1
        return ((1 - s1) // 2) * ((1 - s2) // 2)
    
    def OR(self, a: int, b: int, noise: float = 0.0) -> int:
        """OR gate via phase sum."""
        v = self.encode_inputs(a, b, noise)
        za, zb, _ = self.read_bands(v)
        b1 = self.phase_to_bit(za)
        b2 = self.phase_to_bit(zb)
        return min(1, b1 + b2)
    
    def NOT(self, a: int) -> int:
        """NOT gate - simple bit flip."""
        return 1 - a
    
    def NAND(self, a: int, b: int, noise: float = 0.0) -> int:
        """NAND gate."""
        return self.NOT(self.AND(a, b, noise))
    
    def NOR(self, a: int, b: int, noise: float = 0.0) -> int:
        """NOR gate."""
        return self.NOT(self.OR(a, b, noise))
    
    def XNOR(self, a: int, b: int, noise: float = 0.0) -> int:
        """XNOR gate."""
        return self.NOT(self.XOR(a, b, noise))
    
    def test_gates(self, noise: float = 0.0) -> dict:
        """Test all gates and return accuracy scores."""
        gates = ['XOR', 'AND', 'OR', 'NAND', 'NOR', 'XNOR']
        results = {}
        
        for gate_name in gates:
            gate = getattr(self, gate_name)
            correct = 0
            
            for a in [0, 1]:
                for b in [0, 1]:
                    # Expected output
                    if gate_name == 'XOR':
                        expected = a ^ b
                    elif gate_name == 'AND':
                        expected = a & b
                    elif gate_name == 'OR':
                        expected = a | b
                    elif gate_name == 'NAND':
                        expected = 1 - (a & b)
                    elif gate_name == 'NOR':
                        expected = 1 - (a | b)
                    elif gate_name == 'XNOR':
                        expected = 1 - (a ^ b)
                    
                    # Actual output
                    actual = gate(a, b, noise)
                    
                    if actual == expected:
                        correct += 1
            
            results[gate_name] = correct / 4.0
        
        return results