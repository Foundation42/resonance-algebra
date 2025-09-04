#!/usr/bin/env python3
"""
Resonant Phase Logic — Boolean gates & small circuits via spectral phase algebra.

- Encode bits as phases: b ∈ {0,1} → z = exp(i*pi*b) ∈ {+1, -1}
- Read s = Re(z) ∈ {+1, -1}; recover bit b = (1 - s)/2
- XOR emerges as phase difference parity; AND/OR as simple polynomials in s1, s2
- Half-adder & full-adder built from XOR/AND
- Optional phase noise to test robustness

Usage:
  python resonance_phase_logic.py gates
  python resonance_phase_logic.py adder
  python resonance_phase_logic.py stress --noise 0.2 --trials 1000
"""

import argparse
import numpy as np
from dataclasses import dataclass

rng = np.random.default_rng(123)

@dataclass
class Lens:
    B: np.ndarray  # d x r, orthonormal columns

    @staticmethod
    def orthonormal(d: int, r: int, seed: int = 0) -> "Lens":
        rng_local = np.random.default_rng(seed)
        M = rng_local.normal(size=(d, r))
        Q, _ = np.linalg.qr(M)
        return Lens(Q)

    def project(self, v: np.ndarray) -> np.ndarray:
        return self.B.T @ v

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        return self.B @ coeffs

class PhaseLogic:
    def __init__(self, d: int = 32, r: int = 8, seed: int = 0):
        self.lens = Lens.orthonormal(d, r, seed=seed)
        self.band_a = 0
        self.band_b = 1
        self.band_diff = 2  # interference band encodes XOR/parity

    @staticmethod
    def bit_to_phase(b: int, noise: float = 0.0) -> complex:
        """Map 0→phase 0, 1→phase π, add optional Gaussian noise to phase (radians)."""
        base = 0.0 if b == 0 else np.pi
        phi = base + noise * rng.normal()
        return np.exp(1j * phi)

    @staticmethod
    def phase_to_s(z: complex) -> int:
        """Return s ∈ {+1,-1} from complex unit z by real-part sign (robust)."""
        return 1 if np.real(z) >= 0 else -1

    @staticmethod
    def s_to_bit(s: int) -> int:
        """Map s ∈ {+1,-1} to bit b ∈ {0,1}."""
        return (1 - s) // 2

    def encode_pair(self, a: int, b: int, noise: float = 0.0) -> np.ndarray:
        """Encode two bits into spectral coeffs: bands [a, b, a-b]."""
        r = self.lens.B.shape[1]
        coeffs = np.zeros(r, dtype=complex)
        za = self.bit_to_phase(a, noise)
        zb = self.bit_to_phase(b, noise)
        coeffs[self.band_a] = za
        coeffs[self.band_b] = zb
        coeffs[self.band_diff] = za * np.conj(zb)  # phase difference
        # Optional: sprinkle tiny energy in other bands for realism
        return self.lens.reconstruct(coeffs)

    def read_phases(self, v: np.ndarray) -> tuple[complex, complex, complex]:
        s = self.lens.project(v)
        return s[self.band_a], s[self.band_b], s[self.band_diff]

    # ---- Gates in phase/s-domain ----

    def XOR(self, a: int, b: int, noise: float = 0.0) -> int:
        v = self.encode_pair(a, b, noise=noise)
        _, _, zdiff = self.read_phases(v)
        # XOR true if phase difference near π (real part negative)
        s_xor = -1 if np.real(zdiff) < 0 else 1  # s for XOR (±1)
        return self.s_to_bit(s_xor)

    def NOT(self, a: int) -> int:
        return 1 - a

    def OR(self, a: int, b: int, noise: float = 0.0) -> int:
        # Use s-domain polynomial: b = (1-s)/2; OR = b1 + b2 - b1*b2
        v = self.encode_pair(a, b, noise=noise)
        za, zb, _ = self.read_phases(v)
        s1, s2 = self.phase_to_s(za), self.phase_to_s(zb)
        b1, b2 = self.s_to_bit(s1), self.s_to_bit(s2)
        return int(b1 + b2 - b1*b2)

    def AND(self, a: int, b: int, noise: float = 0.0) -> int:
        v = self.encode_pair(a, b, noise=noise)
        za, zb, _ = self.read_phases(v)
        s1, s2 = self.phase_to_s(za), self.phase_to_s(zb)
        # AND = ((1 - s1)/2) * ((1 - s2)/2)
        return int(((1 - s1)//2) * ((1 - s2)//2))

    def NAND(self, a: int, b: int, noise: float = 0.0) -> int:
        return 1 - self.AND(a,b,noise=noise)

    def half_adder(self, a: int, b: int, noise: float = 0.0) -> tuple[int,int]:
        s = self.XOR(a,b,noise=noise)
        c = self.AND(a,b,noise=noise)
        return s, c

    def full_adder(self, a: int, b: int, cin: int, noise: float = 0.0) -> tuple[int,int]:
        s1, c1 = self.half_adder(a,b,noise=noise)
        s2, c2 = self.half_adder(s1,cin,noise=noise)
        cout = 1 if (c1 + c2) > 0 else 0
        return s2, cout

def demo_gates(noise: float=0.0):
    pl = PhaseLogic()
    print("Truth tables (noise =", noise, ")")
    print("A B | XOR AND OR NAND")
    for a in (0,1):
        for b in (0,1):
            print(a, b, "|",
                  pl.XOR(a,b,noise),
                  pl.AND(a,b,noise),
                  pl.OR(a,b,noise),
                  pl.NAND(a,b,noise))

def demo_adder(noise: float=0.0):
    pl = PhaseLogic()
    print("Half-adder: (A,B) -> (Sum, Carry)")
    for a in (0,1):
        for b in (0,1):
            print(f"({a},{b}) -> {pl.half_adder(a,b,noise)}")
    print("\nFull-adder: (A,B,Cin) -> (Sum, Cout)")
    for a in (0,1):
        for b in (0,1):
            for cin in (0,1):
                print(f"({a},{b},{cin}) -> {pl.full_adder(a,b,cin,noise)}")

def stress(noise: float=0.2, trials: int=1000):
    pl = PhaseLogic()
    ok_xor = ok_and = ok_or = 0
    for _ in range(trials):
        a,b = rng.integers(0,2), rng.integers(0,2)
        if pl.XOR(a,b,noise)==(a^b): ok_xor += 1
        if pl.AND(a,b,noise)==(a&b): ok_and += 1
        if pl.OR(a,b,noise)==(a|b): ok_or += 1
    print(f"Noise σ={noise} rad | XOR: {ok_xor/trials:.3f}  AND: {ok_and/trials:.3f}  OR: {ok_or/trials:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=["gates","adder","stress"])
    ap.add_argument("--noise", type=float, default=0.0, help="stddev of phase noise (radians)")
    ap.add_argument("--trials", type=int, default=1000)
    args = ap.parse_args()

    if args.demo=="gates":
        demo_gates(args.noise)
    elif args.demo=="adder":
        demo_adder(args.noise)
    else:
        stress(args.noise, args.trials)

if __name__ == "__main__":
    main()
