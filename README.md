# Vector Perturbation Precoding (VPP) via QUBO

This repository contains an experimental implementation of **downlink multi user Vector Perturbation Precoding (VPP)**, its formulation as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem, and numerical evaluation using both combinatorial optimization metrics and communication performance metrics.

The goals of this work are to study:

- How the **QUBO structure** changes with the spacing parameter \( \tau \)
- How difficult the resulting QUBO is to solve
- How \( \tau \) affects **bit error rate (BER)** at the receiver

The implementation prioritizes clarity and interpretability over computational efficiency.

---

## System Model

- Downlink MU MIMO system  
- Number of users: `Nr = 4`  
- Number of transmit antennas: `Nt = 4`  
- Modulation: **QPSK**, with comparison to **16QAM**  
- Channel model: i.i.d. Rayleigh fading  
- Precoding: **Zero Forcing (ZF)**  

The received signal model is

\[
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
\]

With ZF precoding

\[
\mathbf{x} =
\frac{\mathbf{P}(\mathbf{u} + \tau \mathbf{v})}
{\left\lVert \mathbf{P}(\mathbf{u} + \tau \mathbf{v}) \right\rVert},
\qquad
\mathbf{P} = \mathbf{H}^\dagger
\]

---

## Vector Perturbation Precoding (VPP)

- The perturbation vector
\[
\mathbf{v} \in \mathbb{Z}^N + j\,\mathbb{Z}^N
\]
is chosen to **minimize transmit power**

- Each real and imaginary component is restricted to
\[
\{-1, 0, 1\}
\]

- For QPSK, the conventional spacing parameter is
\[
\tau = 4
\]

---

## QUBO Formulation

### Variable Count

- 4 users  
- 4 complex perturbation entries  
- Real and imaginary parts → 8 integer variables  
- Each integer encoded using **2 binary variables**  
- Total QUBO variables: **16**  
- QUBO matrix size: **16 × 16**

### Encoding \(\{-1,0,1\}\) with Two Bits

- Two bits give four states, but only three are valid  
- Valid states correspond to \( -1, 0, +1 \)  
- The remaining state is assigned a **penalty energy**  
- Invalid combinations are avoided automatically during optimization  

---

## QUBO Visualization

Two heatmap visualizations are generated for each QUBO.

### 1. Global Normalization

- Plot \( |Q| / \max(|Q|) \)
- Shows relative magnitude of all coefficients
- Diagonal entries correspond to **linear terms**
- Off diagonal entries correspond to **quadratic couplings**

### 2. Off Diagonal Normalization

- Diagonal entries set to zero
- Plot \( |Q_{ij}| / \max(|Q|) \) for \( i \neq j \)
- Highlights the **coupling structure only**

### Diagonal vs Off Diagonal Metrics

For each QUBO, the following metrics are computed:

- `max_diag`
- `max_offdiag`
- `ratio_max_off_to_diag`
- `ratio_mean_off_to_diag`

These quantify how strongly coupled the optimization problem is.

---

## Observed QUBO Behavior

### QPSK Detection (Baseline)

- Strong diagonal dominance
- Weak off diagonal terms
- Typical ratio
\[
\frac{\max |Q_{ij}|}{\max |Q_{ii}|} \approx 0.27
\]

### VPP with QPSK

As \( \tau \) increases, coupling strength increases:

- \( \tau = 4 \) → ratio \( \approx 0.72 \)
- \( \tau = 8 \) → ratio \( \approx 0.84 \)
- \( \tau = 12 \) → ratio \( \approx 0.89 \)

VPP QUBOs are therefore significantly more coupled than standard QPSK detection QUBOs.

### VPP with 16QAM

- Even stronger off diagonal dominance
- Confirms that higher order constellations produce harder QUBOs

---

## Solver Experiments

### Ground State Hit Rate vs \( \tau \)

- Small \( \tau \) → very low hit rate
- Large \( \tau \) → near perfect hit rate

Larger \( \tau \) simplifies the optimization landscape, but does not guarantee good communication performance.

---

## BER Simulation

### Receiver Processing

1. Undo transmit power normalization  
2. Apply modulo-\( \tau \) operation  
3. Nearest neighbor slicing  

### Observed BER Trends

- \( \tau = 1, 2 \) → BER \( \approx 0.5 \) (random guessing)
- \( \tau = 4 \) → sharp BER decay with SNR
- \( \tau = 8, 12 \) → good BER with diminishing returns

Very small \( \tau \) can result in near zero transmit power and numerical instability. These cases are safely skipped.

---

## Baseline Comparison

A **ZF only (no VPP)** BER curve is included as a reference.  
This allows direct evaluation of the gain provided by VPP.
---
