# Vector Perturbation Precoding via QUBO (PySA)

This repository implements and analyzes **Vector Perturbation Precoding (VPP)** for downlink multi-user MIMO using a **QUBO (Quadratic Unconstrained Binary Optimization)** formulation.  
The focus is on understanding the role of the **spacing parameter τ (tau)**, the **structure of the QUBO matrix**, and the **bit-error-rate (BER) performance** compared with conventional baselines.

The work is motivated by discussions on whether QUBO-based VPP behaves similarly to conventional VPP and MIMO detection, especially for **QPSK**.

---

## Implemented Components

### 1. QUBO Construction for VPP
- Downlink MU-MIMO system with ZF precoding.
- VPP perturbation vector  
  \[
  \mathbf{v} \in \mathbb{Z}[j]^K
  \]
  with
  \[
  \Re(v_k), \Im(v_k) \in \{-1,0,1\}
  \]
- Each real and imaginary component is encoded using **two binary variables**, resulting in:
  - 4 binary variables per user
  - 16 binary variables total for a 4×4 system
- This produces a **16×16 QUBO matrix**.

Scripts:
- `qubo_from_vpp(...)`
- `run_vpp_qubo_visualization.py`

---

### 2. QUBO Structure Visualization
To understand why QPSK VPP is harder than MIMO detection, the QUBO matrix is visualized using heatmaps.

Two normalizations are used:

#### (a) Global Normalized Heatmap
