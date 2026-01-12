import numpy as np
from vpp_qubo.vpp import sample_and_build, brute_force_check

#ex 1

#random VPP instance 1
H, u, tau, data = sample_and_build(seed=3)

#H = random channel matrix 2x2 
#u=random user symbols QPSK
#tau set to 4.0 in sample_and_build
#data is dictionary with QUBO (Q, offset, C, and decode)


# Checks QUBO correctness with brute force
# searches all possible integer perturbations l, searches all possible binary vectors q (QUBO method), and then compares best energies and perturbations

res = brute_force_check(H, u, tau, data["Q"], data["offset"], data["C"], data["decode"])

print("\n")
print("Example 1: ")
print("Channel matrix H:\n", H)
print("User symbols u:", u)
print("Tau:", tau)
print("QUBO matrix Q:\n", data["Q"])
print("Offset:", data["offset"])
print("Direct best energy:", res["E_direct"])
print("QUBO best energy:", res["E_qubo"])
print("Decoded perturbation vector l* from QUBO:", res["l_from_qubo"])
print("Decoded perturbation vector l* direct:", res["l_direct"])

#1e-8 arbitarily chosen as tolerance
print("Match:", abs(res["E_direct"] - res["E_qubo"]) < 1e-8)

#ex 2

#random VPP instance 2
H2, u2, tau2, data2 = sample_and_build(seed=4)
res2 = brute_force_check(H2, u2, tau2, data2["Q"], data2["offset"], data2["C"], data2["decode"])

print("\n")
print("Example 2: ")
print("Channel matrix H:\n", H2)
print("User symbols u:", u2)
print("Tau:", tau2)
print("QUBO matrix Q:\n", data2["Q"])
print("Offset:", data2["offset"])
print("Direct best energy:", res2["E_direct"])
print("QUBO best energy:", res2["E_qubo"])
print("Decoded perturbation vector l* from QUBO:", res2["l_from_qubo"])
print("Decoded perturbation vector l* direct:", res2["l_direct"])
print("Match:", abs(res2["E_direct"] - res2["E_qubo"]) < 1e-8)