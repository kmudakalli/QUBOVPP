# Vector Perturbation Precoding via QUBO Formulation

This repository implements the mapping of the **Vector Perturbation Precoding (VPP)** problem in MIMO systems into a **Quadratic Unconstrained Binary Optimization (QUBO)** form.  
The code demonstrates how to represent complex-valued precoding with integer perturbations as a binary optimization problem suitable for quantum or classical annealing solvers.

---

## File Overview

### 1. `encoding.py`
This module handles **integer-to-binary encoding** using two's complement representation.

- **`complex_to_real(M)`**  
  Converts complex vectors or matrices into their real-valued equivalents by stacking real and imaginary parts.  
  - Vectors: `[Re(u); Im(u)]`  
  - Matrices:  
    \[
    \Phi(P) = \begin{bmatrix}
    \Re(P) & -\Im(P) \\
    \Im(P) &  \Re(P)
    \end{bmatrix}
    \]

- **`build_integer_encoding(n_vars, t)`**  
  Builds a binary encoding matrix `C` for integer variables using `t` magnitude bits (total `t+1` bits per variable).  
  Also returns a decode function to map binary vectors back to integer values.  
  - Example: `t=1` → weights `[-2, 1]` → values in `{−2, −1, 0, 1}`.

---

### 2. `vpp.py`
This module defines the **Vector Perturbation Precoding (VPP)** problem in both its complex and realified forms.

- Constructs the channel, user symbols, and precoding matrix.  
- Converts complex-valued models to real form using `complex_to_real` from `encoding.py`.  
- Defines:
  \[
  E(v) = \|F(y + \tau v)\|_2^2
  \]
  where \( F = \Phi(P) \), \( y = \phi(u) \), and \( G = F^T F \).  
- Provides a function to build the equivalent **QUBO matrix**:
  \[
  Q = \tau^2 C^T G C + 2\tau C^T G y
  \]
  and a scalar offset term.

This module is the **core mathematical engine** of the project.

---

### 3. `vpp_qubo_example.py`
This is the **demonstration script** tying everything together.

It:
1. Creates example MIMO channels and user symbols.
2. Runs the VPP-to-QUBO conversion using `vpp.py`.
3. Evaluates the energy both directly and through the QUBO form.
4. Compares results to confirm that:
   - The QUBO energy matches the direct energy.
   - The decoded perturbation vector is consistent.
   - Prints all matrices, offsets, and verification outcomes.

Output looks like:
