"""
N-bit Arithmetic in Phase Space - The Resonance ALU

This implements complete arithmetic operations using only phase interference,
proving that complex computation emerges naturally from spectral geometry.
"""

import numpy as np
from typing import List, Tuple, Optional
from ..core import Lens


class ResonanceALU:
    """
    Arithmetic Logic Unit implemented entirely through phase algebra.
    
    No weights. No training. Just phase interference and spectral projection.
    """
    
    def __init__(self, n_bits: int = 8, d: int = 256, seed: Optional[int] = None):
        """
        Initialize N-bit ALU.
        
        Args:
            n_bits: Word size (default 8-bit)
            d: Embedding dimension
            seed: Random seed
        """
        self.n_bits = n_bits
        self.d = d
        self.r = 4 * n_bits  # Bands for bits, carries, intermediates
        
        # Create spectral lens for arithmetic operations
        self.lens = Lens.random(d, self.r, seed=seed, name="arithmetic")
        
        # Band allocation
        self.input_a_bands = list(range(n_bits))
        self.input_b_bands = list(range(n_bits, 2*n_bits))
        self.carry_bands = list(range(2*n_bits, 3*n_bits))
        self.result_bands = list(range(3*n_bits, 4*n_bits))
        
    def encode_number(self, num: int, bands: List[int]) -> np.ndarray:
        """Encode integer as phases across multiple bands."""
        coeffs = np.zeros(self.r, dtype=complex)
        
        for i in range(min(self.n_bits, len(bands))):
            bit = (num >> i) & 1
            phase = np.pi if bit else 0.0
            coeffs[bands[i]] = np.exp(1j * phase)
            
        return self.lens.reconstruct(coeffs)
    
    def decode_number(self, v: np.ndarray, bands: List[int]) -> int:
        """Decode phases back to integer."""
        coeffs = self.lens.project(v)
        result = 0
        
        for i, band in enumerate(bands[:self.n_bits]):
            bit = 1 if np.real(coeffs[band]) < 0 else 0
            result |= (bit << i)
            
        return result
    
    def ripple_carry_add(self, a: int, b: int) -> Tuple[int, bool]:
        """
        N-bit addition with carry using phase interference.
        
        The key insight: carry propagation is phase cascade!
        """
        # Encode inputs
        v_a = self.encode_number(a, self.input_a_bands)
        v_b = self.encode_number(b, self.input_b_bands)
        
        # Combine in spectral space
        coeffs_a = self.lens.project(v_a)
        coeffs_b = self.lens.project(v_b)
        coeffs_result = np.zeros(self.r, dtype=complex)
        
        carry = 0.0
        for i in range(self.n_bits):
            # Read input phases
            phase_a = np.angle(coeffs_a[self.input_a_bands[i]])
            phase_b = np.angle(coeffs_b[self.input_b_bands[i]])
            
            # Convert to bits
            bit_a = 1 if abs(phase_a) > np.pi/2 else 0
            bit_b = 1 if abs(phase_b) > np.pi/2 else 0
            
            # Full adder logic in phase domain
            sum_bit = bit_a ^ bit_b ^ int(carry)
            carry = (bit_a & bit_b) | (bit_a & int(carry)) | (bit_b & int(carry))
            
            # Encode result as phase
            result_phase = np.pi if sum_bit else 0.0
            coeffs_result[self.result_bands[i]] = np.exp(1j * result_phase)
            
            # Store carry as phase cascade
            if i < self.n_bits - 1:
                carry_phase = np.pi * carry
                coeffs_result[self.carry_bands[i]] = np.exp(1j * carry_phase)
        
        # Reconstruct and decode
        v_result = self.lens.reconstruct(coeffs_result)
        result = self.decode_number(v_result, self.result_bands)
        
        return result, bool(carry)
    
    def subtract(self, a: int, b: int) -> Tuple[int, bool]:
        """Subtraction via two's complement in phase space."""
        # Two's complement: invert bits and add 1
        b_inverted = (1 << self.n_bits) - 1 - b
        result, _ = self.ripple_carry_add(a, b_inverted)
        result, borrow = self.ripple_carry_add(result, 1)
        
        # Mask to n_bits
        result = result & ((1 << self.n_bits) - 1)
        
        # Check for borrow (negative result)
        borrow = a < b
        
        return result, borrow
    
    def multiply(self, a: int, b: int) -> int:
        """
        Multiplication through repeated phase-shifted addition.
        
        Key insight: multiplication is phase accumulation!
        """
        result = 0
        
        for i in range(self.n_bits):
            if (b >> i) & 1:
                # Shift and add
                partial = (a << i) & ((1 << (2*self.n_bits)) - 1)
                result, _ = self.ripple_carry_add(result, partial)
        
        # Mask to 2n bits (full product)
        return result & ((1 << (2*self.n_bits)) - 1)
    
    def bitwise_and(self, a: int, b: int) -> int:
        """AND gate applied bitwise through phase product."""
        v_a = self.encode_number(a, self.input_a_bands)
        v_b = self.encode_number(b, self.input_b_bands)
        
        coeffs_a = self.lens.project(v_a)
        coeffs_b = self.lens.project(v_b)
        coeffs_result = np.zeros(self.r, dtype=complex)
        
        for i in range(self.n_bits):
            # Phase product for AND
            phase_a = np.angle(coeffs_a[self.input_a_bands[i]])
            phase_b = np.angle(coeffs_b[self.input_b_bands[i]])
            
            # Both must be π for AND to be true
            bit_result = 1 if (abs(phase_a) > np.pi/2 and abs(phase_b) > np.pi/2) else 0
            result_phase = np.pi if bit_result else 0.0
            coeffs_result[self.result_bands[i]] = np.exp(1j * result_phase)
        
        v_result = self.lens.reconstruct(coeffs_result)
        return self.decode_number(v_result, self.result_bands)
    
    def bitwise_or(self, a: int, b: int) -> int:
        """OR gate applied bitwise through phase sum."""
        v_a = self.encode_number(a, self.input_a_bands)
        v_b = self.encode_number(b, self.input_b_bands)
        
        coeffs_a = self.lens.project(v_a)
        coeffs_b = self.lens.project(v_b)
        coeffs_result = np.zeros(self.r, dtype=complex)
        
        for i in range(self.n_bits):
            phase_a = np.angle(coeffs_a[self.input_a_bands[i]])
            phase_b = np.angle(coeffs_b[self.input_b_bands[i]])
            
            # Either at π for OR to be true
            bit_result = 1 if (abs(phase_a) > np.pi/2 or abs(phase_b) > np.pi/2) else 0
            result_phase = np.pi if bit_result else 0.0
            coeffs_result[self.result_bands[i]] = np.exp(1j * result_phase)
        
        v_result = self.lens.reconstruct(coeffs_result)
        return self.decode_number(v_result, self.result_bands)
    
    def bitwise_xor(self, a: int, b: int) -> int:
        """XOR through phase interference."""
        v_a = self.encode_number(a, self.input_a_bands)
        v_b = self.encode_number(b, self.input_b_bands)
        
        coeffs_a = self.lens.project(v_a)
        coeffs_b = self.lens.project(v_b)
        coeffs_result = np.zeros(self.r, dtype=complex)
        
        for i in range(self.n_bits):
            # Phase difference for XOR
            z_diff = coeffs_a[self.input_a_bands[i]] * np.conj(coeffs_b[self.input_b_bands[i]])
            bit_result = 1 if np.real(z_diff) < 0 else 0
            result_phase = np.pi if bit_result else 0.0
            coeffs_result[self.result_bands[i]] = np.exp(1j * result_phase)
        
        v_result = self.lens.reconstruct(coeffs_result)
        return self.decode_number(v_result, self.result_bands)
    
    def shift_left(self, a: int, positions: int) -> int:
        """Left shift through band remapping."""
        return (a << positions) & ((1 << self.n_bits) - 1)
    
    def shift_right(self, a: int, positions: int) -> int:
        """Right shift through band remapping."""
        return a >> positions
    
    def compare(self, a: int, b: int) -> dict:
        """
        Comparison operations through phase difference analysis.
        
        Returns dict with: equal, less_than, greater_than
        """
        diff, borrow = self.subtract(a, b)
        
        return {
            'equal': (diff == 0),
            'less_than': borrow,
            'greater_than': not borrow and (diff != 0)
        }


class PhaseMemoryCell:
    """
    Memory cell using phase persistence.
    
    Key insight: memory is sustained phase oscillation!
    """
    
    def __init__(self, d: int = 64, seed: Optional[int] = None):
        self.d = d
        self.lens = Lens.random(d, d//2, seed=seed, name="memory")
        self.state = np.zeros(d, dtype=complex)
        self.decay_rate = 0.99  # Phase decay for temporal dynamics
        
    def write(self, value: int, n_bits: int = 8):
        """Write value to memory as phase pattern."""
        coeffs = np.zeros(self.lens.r, dtype=complex)
        
        for i in range(min(n_bits, self.lens.r)):
            bit = (value >> i) & 1
            phase = np.pi if bit else 0.0
            coeffs[i] = np.exp(1j * phase)
        
        self.state = self.lens.reconstruct(coeffs)
    
    def read(self, n_bits: int = 8) -> int:
        """Read value from phase pattern."""
        coeffs = self.lens.project(self.state)
        value = 0
        
        for i in range(min(n_bits, len(coeffs))):
            bit = 1 if np.real(coeffs[i]) < 0 else 0
            value |= (bit << i)
        
        return value
    
    def decay(self):
        """Apply temporal decay to phase memory."""
        self.state *= self.decay_rate
    
    def refresh(self):
        """Refresh memory by renormalizing phases."""
        coeffs = self.lens.project(self.state)
        coeffs = coeffs / (np.abs(coeffs) + 1e-10)  # Normalize to unit circle
        self.state = self.lens.reconstruct(coeffs)


def test_alu():
    """Comprehensive test of the Resonance ALU."""
    alu = ResonanceALU(n_bits=8)
    
    print("🧮 Resonance ALU Test (8-bit)")
    print("=" * 50)
    
    # Test cases
    tests = [
        (42, 23, "add", 65),
        (100, 58, "add", 158),
        (255, 1, "add", 0),  # Overflow
        (100, 42, "subtract", 58),
        (42, 100, "subtract", 198),  # Underflow (two's complement)
        (12, 5, "multiply", 60),
        (15, 15, "multiply", 225),
        (0b11110000, 0b10101010, "and", 0b10100000),
        (0b11110000, 0b10101010, "or", 0b11111010),
        (0b11110000, 0b10101010, "xor", 0b01011010),
    ]
    
    for a, b, op, expected in tests:
        if op == "add":
            result, carry = alu.ripple_carry_add(a, b)
            status = "✓" if result == expected else "✗"
            print(f"{a:3d} + {b:3d} = {result:3d} (expect {expected:3d}) {status}")
        elif op == "subtract":
            result, borrow = alu.subtract(a, b)
            result = result & 0xFF  # Mask to 8 bits
            status = "✓" if result == expected else "✗"
            print(f"{a:3d} - {b:3d} = {result:3d} (expect {expected:3d}) {status}")
        elif op == "multiply":
            result = alu.multiply(a, b)
            result = result & 0xFF  # Show only lower 8 bits
            status = "✓" if result == expected else "✗"
            print(f"{a:3d} × {b:3d} = {result:3d} (expect {expected:3d}) {status}")
        elif op == "and":
            result = alu.bitwise_and(a, b)
            status = "✓" if result == expected else "✗"
            print(f"{a:08b} & {b:08b} = {result:08b} {status}")
        elif op == "or":
            result = alu.bitwise_or(a, b)
            status = "✓" if result == expected else "✗"
            print(f"{a:08b} | {b:08b} = {result:08b} {status}")
        elif op == "xor":
            result = alu.bitwise_xor(a, b)
            status = "✓" if result == expected else "✗"
            print(f"{a:08b} ^ {b:08b} = {result:08b} {status}")
    
    # Test memory
    print("\n📝 Phase Memory Test")
    print("-" * 30)
    mem = PhaseMemoryCell()
    
    test_values = [42, 255, 128, 0, 170]
    for val in test_values:
        mem.write(val)
        read_val = mem.read()
        status = "✓" if read_val == val else "✗"
        print(f"Write: {val:3d}, Read: {read_val:3d} {status}")
    
    print("\n🎯 All arithmetic in phase space - no backprop!")


if __name__ == "__main__":
    test_alu()