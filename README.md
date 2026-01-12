# Vector Perturbation Precoding (VPP) via QUBO

This repository implements and analyzes **Vector Perturbation Precoding (VPP)** for downlink multi-user MIMO using a **QUBO (Quadratic Unconstrained Binary Optimization)** formulation.  
The focus is on understanding the role of the **spacing parameter τ (tau)**, the **structure of the QUBO matrix**, and the **bit-error-rate (BER) performance** compared with conventional baselines.

The work is motivated by discussions on whether QUBO-based VPP behaves similarly to conventional VPP and MIMO detection, especially for **QPSK**.
---

## Setup

- Downlink MU-MIMO
- Users: Nr = 4
- TX antennas: Nt = 4
- Modulation: QPSK (with 16QAM comparison)
- Channel: i.i.d. Rayleigh
- Precoder: Zero-Forcing (ZF)

Signal model:
y = H x + n

ZF precoding:
x = H^† (u + τ v) / || H^† (u + τ v) ||

---

## Vector Perturbation

- Perturbation vector:
  v ∈ Z^N + j Z^N
- Each real/imag component constrained to:
  {-1, 0, 1}
- Conventional QPSK spacing:
  τ = 4

---

## QUBO Formulation

- 4 complex perturbation entries
- Real + imaginary parts → 8 integer variables
- Each integer encoded using 2 bits
- Total binary variables: 16
- QUBO size: 16 × 16

Invalid bit patterns are penalized energetically.

---

## QUBO Structure (Empirical)

Coupling strength is analyzed via heatmaps and simple ratios.

- Baseline QPSK detection:
  strong diagonal dominance  
  max(|Q_ij|) / max(|Q_ii|) ≈ 0.27

- VPP with QPSK:
  coupling increases with τ
  - τ = 4  → ~0.72
  - τ = 8  → ~0.84
  - τ = 12 → ~0.89

- VPP with 16QAM:
  even stronger off-diagonal dominance

Higher-order constellations and larger τ produce harder QUBOs.

---

## Solver Behavior

- Small τ → low ground-state hit rate
- Large τ → near-perfect hit rate

Larger τ simplifies the optimization landscape, but does not imply better communication performance.

---

## BER Results

Receiver processing:
1. Undo transmit power normalization
2. Modulo-τ operation
3. Nearest-neighbor slicing

Observed behavior:
- τ = 1, 2 → BER ≈ 0.5
- τ = 4 → clear BER improvement
- τ = 8, 12 → diminishing returns

A ZF-only (no VPP) baseline is included.

---

## Status

Active research prototype focused on understanding QUBO structure versus communication performance.
