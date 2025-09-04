#!/usr/bin/env python3
"""
Resonance Algebra – Minimal Demo (NumPy-only)

Usage examples:
  python resonance_algebra_demo.py analogy
  python resonance_algebra_demo.py retrieval
  python resonance_algebra_demo.py motion
  python resonance_algebra_demo.py all

This is a compact, dependency-light prototype showing:
- Lenses (orthonormal bases), projection, reconstruction
- Spectral resonance (inner product/coherence with band weights)
- Complex-phase binding/unbinding in band space
- Cross-modal alignment via orthogonal Procrustes
- Demos: king→queen analogy; text→image retrieval; phase-as-motion estimate
"""

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

rng = np.random.default_rng(42)

# -----------------------------
# Utilities
# -----------------------------

def orthonormal_columns(mat: np.ndarray) -> np.ndarray:
    """QR-orthonormalize columns."""
    q, _ = np.linalg.qr(mat)
    return q

def random_orthonormal(d: int) -> np.ndarray:
    return orthonormal_columns(rng.normal(size=(d, d)))

def procrustes_orthogonal(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Find R that minimizes ||X R - Y||_F, with R orthogonal (Procrustes)."""
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    R = U @ Vt
    return R

# -----------------------------
# Lens & Space
# -----------------------------

@dataclass
class Lens:
    name: str
    B: np.ndarray  # d x r, orthonormal columns

    def project(self, v: np.ndarray) -> np.ndarray:
        return self.B.T @ v

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        return self.B @ coeffs

@dataclass
class ModalitySpace:
    name: str
    d: int
    lenses: Dict[str, Lens]
    M: np.ndarray

    def project(self, v: np.ndarray, lens: str) -> np.ndarray:
        return self.lenses[lens].project(v)

    def reconstruct(self, coeffs: np.ndarray, lens: str) -> np.ndarray:
        return self.lenses[lens].reconstruct(coeffs)

# -----------------------------
# Concept container
# -----------------------------

@dataclass
class Concept:
    modality: str
    v: np.ndarray  # vector in modality space (real or complex)
    spectra: Dict[str, np.ndarray] = None

# -----------------------------
# Resonance Algebra ops
# -----------------------------

def spectral(v: np.ndarray, lens: Lens) -> np.ndarray:
    return lens.project(v)

def reconstruct(coeffs: np.ndarray, lens: Lens) -> np.ndarray:
    return lens.reconstruct(coeffs)

def resonance(x: Concept, y: Concept, lens: Lens, weights: Optional[np.ndarray]=None, normalize: bool=True) -> Tuple[float, float]:
    """Return (inner, coherence) under a lens and optional band weights."""
    X = spectral(x.v, lens).astype(np.complex128)
    Y = spectral(y.v, lens).astype(np.complex128)
    if weights is None:
        weights = np.ones_like(X, dtype=np.complex128)
    else:
        weights = weights.astype(np.complex128)
    inner = np.sum(weights * X * np.conj(Y))  # Hermitian inner product in band space
    inner_real = float(np.real(inner))
    if not normalize:
        return inner_real, float('nan')
    nx = float(np.sqrt(np.real(np.sum(weights * X * np.conj(X)))))
    ny = float(np.sqrt(np.real(np.sum(weights * Y * np.conj(Y)))))
    coh = 0.0 if (nx==0 or ny==0) else inner_real / (nx * ny)
    return inner_real, coh

def bind_phase(x: Concept, lens: Lens, phase: np.ndarray) -> Concept:
    """Complex-phase binding in band space: rotate selected bands by phase."""
    X = spectral(x.v, lens).astype(np.complex128)
    X_bound = X * np.exp(1j * phase)
    v_new = reconstruct(X_bound, lens)
    return Concept(x.modality, np.real(v_new), spectra=None)

def unbind_phase(x: Concept, lens: Lens, phase: np.ndarray) -> Concept:
    return bind_phase(x, lens, -phase)

def condition(x: Concept, lens: Lens, weights: np.ndarray) -> Concept:
    """Sieve: reweight bands under a lens, then reconstruct."""
    X = spectral(x.v, lens).astype(np.complex128)
    v_new = reconstruct(weights * X, lens)
    return Concept(x.modality, np.real(v_new), spectra=None)

def mix(x: Concept, y: Concept, alpha: float=0.5) -> Concept:
    assert x.modality == y.modality
    return Concept(x.modality, alpha*x.v + (1-alpha)*y.v, spectra=None)

# -----------------------------
# Build toy spaces & lenses
# -----------------------------

d_text = 64
d_img  = 64
r_bands = 16

# Shared latent basis to synthesize concepts
B_latent = orthonormal_columns(rng.normal(size=(d_text, r_bands)))

M_text = random_orthonormal(d_text)
M_img  = random_orthonormal(d_img)

# Lenses
B_text = orthonormal_columns(rng.normal(size=(d_text, r_bands)))
B_img  = orthonormal_columns(rng.normal(size=(d_img,  r_bands)))
lens_text = Lens("topics", B_text)
lens_img  = Lens("shapes", B_img)
space_text = ModalitySpace("text", d_text, {"topics": lens_text}, M_text)
space_img  = ModalitySpace("image", d_img, {"shapes": lens_img}, M_img)

# Band index "legend"
BAND_INDEX = {
    "person": 0,
    "royalty": 1,
    "gender": 2,
    "animal": 3,
    "vehicle": 4,
    "color_red": 5,
    "color_blue": 6,
    "shape_round": 7,
    "pose": 8,
    "texture": 9,
}

def latent_vec(**bands):
    v = np.zeros(r_bands)
    for name, strength in bands.items():
        idx = BAND_INDEX[name]
        v[idx] += strength
    return v

def jitter(x, scale=0.02):
    return x + scale * rng.normal(size=x.shape)

def realize_in_modality(latent_coeffs: np.ndarray, M: np.ndarray) -> np.ndarray:
    v_latent = B_latent @ latent_coeffs
    return M @ jitter(v_latent)

def make_concept(latent_coeffs: np.ndarray, modality: str) -> Concept:
    if modality == "text":
        v = realize_in_modality(latent_coeffs, M_text)
        return Concept("text", v)
    else:
        v = realize_in_modality(latent_coeffs, M_img)
        return Concept("image", v)

# -----------------------------
# Demos
# -----------------------------

def demo_analogy():
    # phases on gender band
    male_phase   = np.zeros(r_bands); male_phase[BAND_INDEX["gender"]]   = +np.pi/4
    female_phase = np.zeros(r_bands); female_phase[BAND_INDEX["gender"]] = -np.pi/4

    person = latent_vec(person=1.0)
    royalty = latent_vec(royalty=1.0)
    gender_amp = latent_vec(gender=1.0)

    king_latent   = person + royalty + gender_amp
    queen_latent  = person + royalty + gender_amp

    king_text  = make_concept(king_latent, "text")
    queen_text = make_concept(queen_latent, "text")

    king_bound  = bind_phase(king_text, lens_text,  male_phase)
    queen_bound = bind_phase(queen_text, lens_text, female_phase)

    # queen' = (king ⊗ male^-1) ⊗ female
    king_unbound = unbind_phase(king_bound, lens_text, male_phase)
    queen_from_king = bind_phase(king_unbound, lens_text, female_phase)

    inner, coh = resonance(queen_from_king, queen_bound, lens_text)
    print("Demo — Analogy via phase binding: king -> queen")
    print(f"Resonance coherence = {coh:.3f} (1.0 is perfect)")

def demo_retrieval():
    pairs_latent = {
        "car_red":  latent_vec(vehicle=1.0) + latent_vec(color_red=0.8),
        "car_blue": latent_vec(vehicle=1.0) + latent_vec(color_blue=0.8),
        "cat_red":  latent_vec(animal=1.0)  + latent_vec(color_red=0.8),
    }
    X = np.vstack([realize_in_modality(lat, M_text) for lat in pairs_latent.values()])
    Y = np.vstack([realize_in_modality(lat, M_img)  for lat in pairs_latent.values()])
    R_text_to_img = procrustes_orthogonal(X, Y)

    def align_text_to_img(v_text: np.ndarray) -> np.ndarray:
        return v_text @ R_text_to_img

    # Query: red car (text) -> image candidates
    query_text = realize_in_modality(pairs_latent["car_red"], M_text)
    query_imgspace = align_text_to_img(query_text)
    query = Concept("image", query_imgspace)

    candidates = {name: Concept("image", realize_in_modality(lat, M_img))
                  for name, lat in pairs_latent.items()}

    W = np.ones(r_bands)
    for k in ("vehicle","color_red","color_blue"):
        W[BAND_INDEX[k]] = 1.3

    scores = {}
    for name, c in candidates.items():
        _, coh = resonance(query, c, lens_img, weights=W, normalize=True)
        scores[name] = coh

    print("Demo — Text→Image retrieval (coherences):")
    for k,v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {k:8s}: {v:.3f}")

def demo_motion():
    pose_band = BAND_INDEX["pose"]

    def estimate_phase_delta(x: Concept, y: Concept, lens: Lens) -> float:
        X = spectral(x.v.astype(np.complex128), lens)
        Y = spectral(y.v.astype(np.complex128), lens)
        return float(np.angle(Y[pose_band] * np.conj(X[pose_band])))

    base_lat = latent_vec(person=1.0) + latent_vec(pose=0.7)
    x0 = make_concept(base_lat, "text")
    delta = +0.6  # radians
    phase_shift = np.zeros(r_bands); phase_shift[pose_band] = delta
    x1 = bind_phase(x0, lens_text, phase_shift)

    est = estimate_phase_delta(x0, x1, lens_text)
    print("Demo — Motion via phase delta in 'pose' band:")
    print(f"True Δphase = {delta:.3f} rad; Estimated Δphase ≈ {est:.3f} rad")

def main():
    parser = argparse.ArgumentParser(description="Resonance Algebra minimal demo")
    parser.add_argument("demo", choices=["analogy","retrieval","motion","all"],
                        help="Which demo to run")
    args = parser.parse_args()

    if args.demo in ("analogy","all"):
        demo_analogy(); print()
    if args.demo in ("retrieval","all"):
        demo_retrieval(); print()
    if args.demo in ("motion","all"):
        demo_motion(); print()

if __name__ == "__main__":
    main()
