# Vector Perturbation Precoding via QUBO Formulation

This repository implements the mapping of the Vector Perturbation Precoding (VPP) problem in MIMO (multiple-input and multiple-output) systems into a Quadratic Unconstrained Binary Optimization (QUBO) form, as referenced in the following research papers attached.

Refer to QUBOFinalSlides.pdf for a detailed explanation

This code demonstrates how to represent complex valued precoding with integer perturbations as a binary optimization problem suitable for quantum or classical annealing solvers.
---

## File Overview

### 1. `encoding.py`
This module handles **integer-to-binary encoding** using two's complement representation.

- **`complex_to_real(M)`**  
  Converts complex vectors or matrices into their real-valued equivalents by stacking real and imaginary parts.  

- **`build_integer_encoding(n_vars, t)`**  
  builds a binary encoding matrix C for integer variables using t magnitude bits (total t+1 bits per variable).  
  also returns a decode function to map binary vectors back to integer values.  

---

### 2. `vpp.py`
This module defines the Vector Perturbation Precoding (VPP) problem in both its complex and realified forms.

- Constructs the channel, user symbols, and precoding matrix.  
- Converts complex-valued models to real form using `complex_to_real` from `encoding.py`.  
- Defines:
  \[
  E(v) = \|F(y + \tau v)\|_2^2
  \]
  where \( F = \Phi(P) \), \( y = \phi(u) \), and \( G = F^T F \).  
- Provides a function to build the equivalent QUBO matrix:
  \[
  Q = \tau^2 C^T G C + 2\tau C^T G y
  \]
  and a scalar offset term.
---

### 3. `vpp_qubo_example.py`
This is the demo script which ties everything together.

It:
1. Creates example MIMO channels and user symbols.
2. Runs the VPP-to-QUBO conversion using vpp.py.
3. Evaluates the energy both directly and through the QUBO form.
4. Compares results to confirm that:
   - The QUBO energy matches the direct energy.
   - The decoded perturbation vector is consistent.
   - Prints all matrices, offsets, and verification outcomes.

Sample output: 

Example 1: 
Channel matrix H:
 [[ 1.44314775-0.32007138j -1.80712807-0.15245022j]
 [ 0.29564053-1.42834589j -0.40147374-0.16400096j]]
User symbols u: [-1.+1.j -1.+1.j]
Tau: 4.0
QUBO matrix Q:
 [[ 1.87529499e+01 -1.05755497e+01 -1.49280287e+01  7.46401436e+00
  -1.09462843e-16  5.47314217e-17  1.58156682e+01 -7.90783412e+00]
 [-1.05755497e+01  6.48684954e+00  7.46401436e+00 -3.73200718e+00
   5.47314217e-17 -2.73657108e-17 -7.90783412e+00  3.95391706e+00]
 [-1.49280287e+01  7.46401436e+00  6.27219344e+01 -2.50000098e+01
  -1.58156682e+01  7.90783412e+00  2.34205635e-16 -1.17102818e-16]
 [ 7.46401436e+00 -3.73200718e+00 -2.50000098e+01  6.13904751e+00
   7.90783412e+00 -3.95391706e+00 -1.17102818e-16  5.85514088e-17]
 [-1.09462843e-16  5.47314217e-17 -1.58156682e+01  7.90783412e+00
   1.56414146e+01 -1.05755497e+01 -1.49280287e+01  7.46401436e+00]
 [ 5.47314217e-17 -2.73657108e-17  7.90783412e+00 -3.95391706e+00
  -1.05755497e+01  8.04261720e+00  7.46401436e+00 -3.73200718e+00]
 [ 1.58156682e+01 -7.90783412e+00  2.34205635e-16 -1.17102818e-16
  -1.49280287e+01  7.46401436e+00  4.51859389e+01 -2.50000098e+01]
 [-7.90783412e+00  3.95391706e+00 -1.17102818e-16  5.85514088e-17
   7.46401436e+00 -3.73200718e+00 -2.50000098e+01  1.49070452e+01]]
Offset: 1.290470672027967
Direct best energy: 1.290470672027967
QUBO best energy: 1.290470672027967
Decoded perturbation vector l* from QUBO: [0 0 0 0]
Decoded perturbation vector l* direct: [0 0 0 0]
Match: True


Example 2: 
Channel matrix H:
 [[-0.46088594-1.16064316j -0.12354378-0.00367926j]
 [ 1.17643052-0.44085544j  0.46608784+0.10509836j]]
User symbols u: [-1.+1.j -1.-1.j]
Tau: 4.0
QUBO matrix Q:
 [[ 2.93803688e+02 -1.40341844e+02  1.37486401e+01 -6.87432005e+00
   1.19306832e-14 -5.96534159e-15 -2.41952331e+02  1.20976166e+02]
 [-1.40341844e+02  6.36109225e+01 -6.87432005e+00  3.43716003e+00
  -5.96534159e-15  2.98267080e-15  1.20976166e+02 -6.04880829e+01]
 [ 1.37486401e+01 -6.87432005e+00  2.48779189e+02 -1.22332045e+02
   2.41952331e+02 -1.20976166e+02 -1.40049659e-14  7.00248294e-15]
 [-6.87432005e+00  3.43716003e+00 -1.22332045e+02  5.91084726e+01
  -1.20976166e+02  6.04880829e+01  7.00248294e-15 -3.50124147e-15]
 [ 1.19306832e-14 -5.96534159e-15  2.41952331e+02 -1.20976166e+02
   2.74438009e+02 -1.40341844e+02  1.37486401e+01 -6.87432005e+00]
 [-5.96534159e-15  2.98267080e-15 -1.20976166e+02  6.04880829e+01
  -1.40341844e+02  7.32937617e+01 -6.87432005e+00  3.43716003e+00]
 [-2.41952331e+02  1.20976166e+02 -1.40049659e-14  7.00248294e-15
   1.37486401e+01 -6.87432005e+00  2.41904869e+02 -1.22332045e+02]
 [ 1.20976166e+02 -6.04880829e+01  7.00248294e-15 -3.50124147e-15
  -6.87432005e+00  3.43716003e+00 -1.22332045e+02  6.25456326e+01]]
Offset: 1.295097338418922
Direct best energy: 1.295097338418922
QUBO best energy: 1.295097338418922
Decoded perturbation vector l* from QUBO: [0 0 0 0]
Decoded perturbation vector l* direct: [0 0 0 0]
Match: True
