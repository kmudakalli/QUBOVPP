# Vector Perturbation Precoding (VPP) via QUBO — Current Implementation

This repository contains an experimental implementation of **downlink multi-user Vector Perturbation Precoding (VPP)**, its formulation as a **QUBO**, and numerical evaluation using both combinatorial optimization and conventional communication metrics.

The focus is on understanding:
- how the **QUBO structure** changes with the spacing parameter τ,
- how difficult the resulting QUBO is to solve,
- and how τ affects **bit-error rate (BER)** at the receiver.

The code is written for clarity and research exploration rather than efficiency.

---

## System model

- Downlink MU-MIMO system
- Number of users: `Nr = 4`
- Number of transmit antennas: `Nt = 4`
- Modulation: **QPSK** (and comparison with 16QAM)
- Channel model: i.i.d. Rayleigh fading
- Precoding: **Zero-Forcing (ZF)**

The received signal model is
\[
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
\]

with ZF precoding
\[
\mathbf{x} = \frac{\mathbf{P}(\mathbf{u} + \tau \mathbf{v})}{\|\mathbf{P}(\mathbf{u} + \tau \mathbf{v})\|}
\quad\text{where}\quad
\mathbf{P} = \mathbf{H}^\dagger
\]

---

## Vector Perturbation Precoding (VPP)

- The perturbation vector  
  \[
  \mathbf{v} \in \mathbb{Z}^N + j\,\mathbb{Z}^N
  \]
  is chosen to **minimize transmit power**.
- In practice, each real/imaginary component is restricted to  
  \[
  \{-1, 0, 1\}
  \]
- For QPSK, the conventional spacing is
  \[
  \tau = 4
  \]

---

## QUBO formulation

### Variable count
- 4 users → 4 complex perturbation entries
- Real + imaginary parts → 8 integer variables
- Each integer encoded with **2 binary variables**
- Total QUBO variables: **16**
- QUBO matrix size: **16 × 16**

### Encoding `{−1,0,1}` with 2 bits
Two bits give four states, but only three are used:
- Valid states represent −1, 0, +1
- The remaining state is **energetically penalized**
- During optimization, invalid combinations are automatically avoided

---

## QUBO visualization

Two types of heatmaps are generated:

### 1. Global normalization
- Plot `|Q| / max(|Q|)`
- Shows **relative strength of all coefficients**
- Diagonal entries correspond to **linear terms**
- Off-diagonal entries correspond to **quadratic couplings**

### 2. Off-diagonal global normalization
- Diagonal entries set to zero
- Plot `|Q_ij| / max(|Q|)` for `i ≠ j`
- Highlights **coupling structure only**

### Diagonal vs off-diagonal metrics
For each QUBO:
- `max_diag`
- `max_offdiag`
- `ratio_max_off_to_diag`
- `ratio_mean_off_to_diag`

These quantify how strongly coupled the problem is.

---

## Observed QUBO behavior

### QPSK MIMO detection (baseline)
- Strong diagonal dominance
- Off-diagonal terms much weaker
- Typical `max_off / max_diag ≈ 0.27`

### VPP-QPSK
As τ increases, coupling strength increases:
- τ = 4 → ratio ≈ 0.72
- τ = 8 → ratio ≈ 0.84
- τ = 12 → ratio ≈ 0.89

This shows that **VPP-QPSK QUBOs are significantly more coupled** than standard QPSK detection.

### VPP-16QAM
- Off-diagonal dominance is even stronger
- Confirms that higher-order constellations produce harder QUBOs

---

## Solver experiments

### Hit-rate vs τ (QUBO ground state)
- Small τ → very low hit rate
- Large τ → near-perfect hit rate

This shows that large τ simplifies the optimization landscape but does not guarantee good communication performance.

---

## BER simulation (user-side)

### Receiver processing
1. Undo power normalization
2. Apply modulo-τ operation
3. Nearest-neighbor slicing

### BER trends observed
- τ = 1, 2 → BER ≈ 0.5 (random guessing)
- τ = 4 → sharp BER decay with SNR
- τ = 8, 12 → good BER, diminishing returns

Very small τ can cause **near-zero transmit power**, leading to numerical instability. These cases are skipped safely.

---

## Baseline comparison
A reference **ZF-only (no VPP)** BER curve is included for comparison. This allows direct evaluation of the gain provided by VPP.

---

## Runtime notes
- QUBO visualization: seconds
- Hit-rate sweeps: minutes
- BER sweeps (brute-force VPP baseline): **tens of minutes** on an Apple M2 Pro

---

## Status
This repository reflects an active research prototype. The implementation prioritizes transparency and interpretability of results over speed.

