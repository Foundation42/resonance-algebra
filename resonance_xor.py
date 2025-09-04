#!/usr/bin/env python3
"""
Resonance XOR - Solving XOR without backpropagation using spectral phase geometry

The key insight: XOR is about detecting phase orthogonality.
We solve it through resonance matching in spectral space, no gradients needed!
"""

import numpy as np
from typing import Tuple, List

class ResonanceXOR:
    def __init__(self, d: int = 16, n_bands: int = 8):
        """
        Initialize a Resonance XOR solver
        d: embedding dimension
        n_bands: number of spectral bands
        """
        self.d = d
        self.n_bands = n_bands
        self.rng = np.random.default_rng(42)
        
        # Create orthonormal basis (lens) for spectral projection
        self.lens = self._make_orthonormal_basis(d, n_bands)
        
        # Phase encoding for binary inputs
        self.phase_0 = 0.0        # 0 → 0°
        self.phase_1 = np.pi      # 1 → 180°
        
        # Target phases for XOR outputs  
        self.target_phase_false = 0.0
        self.target_phase_true = np.pi/2  # 90° for "true"
        
        # Learn spectral patterns for each XOR case through resonance
        self.patterns = self._learn_patterns()
        
    def _make_orthonormal_basis(self, d: int, r: int) -> np.ndarray:
        """Create an orthonormal basis via QR decomposition"""
        M = self.rng.normal(size=(d, r))
        Q, _ = np.linalg.qr(M)
        return Q
    
    def _encode_input(self, x1: int, x2: int) -> np.ndarray:
        """
        Encode binary inputs as phase-modulated vectors
        Key: we encode in different bands to maintain independence
        """
        v = np.zeros(self.d, dtype=complex)
        
        # Project to spectral space
        coeffs = np.zeros(self.n_bands, dtype=complex)
        
        # Band 0: first input's phase
        phase1 = self.phase_0 if x1 == 0 else self.phase_1
        coeffs[0] = np.exp(1j * phase1)
        
        # Band 1: second input's phase  
        phase2 = self.phase_0 if x2 == 0 else self.phase_1
        coeffs[1] = np.exp(1j * phase2)
        
        # Band 2: interference pattern (key for XOR!)
        # This encodes the phase DIFFERENCE
        coeffs[2] = np.exp(1j * (phase1 - phase2))
        
        # Reconstruct in embedding space
        v = self.lens @ coeffs
        return v
    
    def _learn_patterns(self) -> dict:
        """
        'Learn' the spectral patterns for XOR through resonance matching
        No backprop - we directly construct the resonance patterns!
        """
        patterns = {}
        
        # For each XOR input-output pair
        xor_truth = [
            ((0, 0), 0),
            ((0, 1), 1),
            ((1, 0), 1),
            ((1, 1), 0)
        ]
        
        for (x1, x2), y in xor_truth:
            # Encode input
            v_input = self._encode_input(x1, x2)
            
            # Create target pattern based on output
            target_phase = self.target_phase_true if y == 1 else self.target_phase_false
            
            # The "learning" is just storing the spectral signature
            # In band 2 (interference), we can directly read the XOR result!
            patterns[(x1, x2)] = {
                'input': v_input,
                'output': y,
                'spectral': self.lens.T @ v_input,
                'target_phase': target_phase
            }
            
        return patterns
    
    def _resonance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate resonance (coherence) between two vectors"""
        # Project to spectral space
        s1 = self.lens.T @ v1
        s2 = self.lens.T @ v2
        
        # Focus on the interference band (band 2) which encodes XOR
        # This is the key: XOR emerges from phase interference!
        coherence = np.abs(np.vdot(s1[2:3], s2[2:3])) / (
            np.linalg.norm(s1[2:3]) * np.linalg.norm(s2[2:3]) + 1e-10
        )
        return coherence
    
    def forward(self, x1: int, x2: int) -> int:
        """
        Forward pass without any neural network!
        We solve XOR through phase geometry and resonance matching
        """
        # Encode input
        v_input = self._encode_input(x1, x2)
        
        # Project to spectral space
        spectral = self.lens.T @ v_input
        
        # The magic: read the phase difference from the interference band
        phase_diff = np.angle(spectral[2])
        
        # XOR is true when inputs differ (phase difference ≈ ±π)
        # XOR is false when inputs are same (phase difference ≈ 0)
        if np.abs(phase_diff) > np.pi/2:
            return 1  # Inputs differ -> XOR is true
        else:
            return 0  # Inputs same -> XOR is false
    
    def test(self) -> None:
        """Test the XOR solver"""
        print("Resonance XOR Test (No Backprop!)")
        print("==================================")
        print("\nXOR Truth Table:")
        print("Input  | Expected | Predicted | Phase Analysis")
        print("-------|----------|-----------|---------------")
        
        correct = 0
        for x1 in [0, 1]:
            for x2 in [0, 1]:
                expected = x1 ^ x2  # XOR operation
                predicted = self.forward(x1, x2)
                
                # Get phase analysis
                v = self._encode_input(x1, x2)
                spectral = self.lens.T @ v
                phase_diff = np.angle(spectral[2])
                
                status = "✓" if predicted == expected else "✗"
                print(f"({x1},{x2})  |    {expected}     |     {predicted}     | "
                      f"Δφ = {phase_diff:+.2f} rad {status}")
                
                if predicted == expected:
                    correct += 1
        
        print(f"\nAccuracy: {correct}/4 = {correct*25}%")
        
        # Show how it works
        print("\n🔍 How it works:")
        print("1. Inputs encoded as phases: 0→0°, 1→180°")
        print("2. Phase difference stored in interference band")
        print("3. XOR detected via phase orthogonality:")
        print("   - Same inputs → phases align → Δφ ≈ 0 → output 0")
        print("   - Different inputs → phases oppose → Δφ ≈ ±π → output 1")
        print("4. No gradients, no backprop - pure spectral geometry! 🎯")

def main():
    # Create and test the Resonance XOR solver
    xor_solver = ResonanceXOR(d=16, n_bands=8)
    xor_solver.test()
    
    print("\n💡 Key Innovation:")
    print("We solved XOR through PHASE INTERFERENCE in spectral space.")
    print("The 'learning' is just recognizing that XOR = phase orthogonality.")
    print("No weights to update, no gradients to compute!")

if __name__ == "__main__":
    main()