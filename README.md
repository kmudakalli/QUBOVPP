# Vector Perturbation Precoding (VPP) via QUBO

The following is work done in collaboration and with the guidance of [Professor Minsung Kim](https://sites.google.com/view/minsungk/home)

QUBOVPP implements **Vector Perturbation Precoding (VPP)** for downlink multi-user MIMO using a **QUBO (Quadratic Unconstrained Binary Optimization)** formulation. It follows prior work on quantum and physics-inspired optimization for wireless precoding.

The current focus is on understanding the role of the **spacing parameter τ (tau)**, the **structure of the QUBO matrix**, and the **bit-error-rate (BER) performance** compared with conventional baselines.

This work is primarily motivated by:
- [S. Kasi et al., *Quantum Annealing for Large MIMO Downlink Vector Perturbation Precoding*, IEEE ICC 2021.](2102.12540v1.pdf)
- [S. Winter et al., *A Lattice-Reduction Aided Vector Perturbation Precoder Relying on Quantum Annealing*, IEEE Wireless Communications Letters, 2024.](A_Lattice-Reduction_Aided_Vector_Perturbation_Precoder_Relying_on_Quantum_Annealing.pdf)

Related optimization and formulation techniques are informed by earlier work on QUBO models for large MIMO processing as well as inspirations for visuals depicted here:
- [M. Kim et al., *Leveraging Quantum Annealing for Large MIMO Processing in Centralized Radio Access Networks*, SIGCOMM 2019.](3341302.3342072.pdf)
- [M. Kim et al., *Physics-Inspired Heuristics for Soft MIMO Detection in 5G New Radio and Beyond*, MobiCom 2021.](3447993.3448619.pdf)

The current focus of this repository is on:
- the role of the spacing parameter τ (tau),
- the resulting **QUBO matrix structure and coupling strength**,
- and **bit-error-rate (BER)** performance compared with conventional zero-forcing and non-VPP baselines.
---

See [QUBOSlides](QUBOSlides.pdf) for more details
