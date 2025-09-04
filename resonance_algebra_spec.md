
# Resonance Algebra (Minimal Spec)

**Objects**
- Modality spaces \(\mathbb{R}^{d_m}\), lenses \(B_m^{(k)}\) with orthonormal columns.
- Concepts \(x_m \in \mathbb{R}^{d_m}\) (optionally complex).
- Alignment maps \(A_{a\to b}\) (orthogonal for simplicity).

**Core Ops**
- Projection: \(\hat x = B^\top x\).
- Resonance: \(\langle x,y\rangle_{B,W} = \sum_i W_i \hat x_i \hat y_i\); coherence = normalized inner product.
- Binding: in band space, \(\hat x' = \hat x \odot e^{i\phi}\); unbind with \(-\phi\).
- Conditioning (sieve): \(\hat x' = W \odot \hat x\).
- Alignment: \(x_b = x_a A_{a\to b}\).

**Demos**
- Word2Vec-style analogy becomes unbind/bind in a demographic band.
- Cross-modal retrieval: project aligned text into image space; compare spectra under an image lens with band weights.
- Motion: phase deltas in a pose band act like motion vectors; estimate via \(\angle(\hat y \cdot \overline{\hat x})\).
