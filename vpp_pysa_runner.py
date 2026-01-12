# ==============================================================
# Quantum-inspired Vector Perturbation Precoding (VPP) via PySA
# with SA / PT sweeps and downlink BER simulation
# ==============================================================

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from itertools import product
from pysa.sa import Solver
from numpy.random import default_rng


from vpp_qubo.vpp import (
    sample_and_build,
    brute_force_check,
    qubo_from_vpp
)
from vpp_analysis_utils import draw_symbols



# --------------------------------------------------------------
# 1. Generate a random VPP instance and its QUBO
# --------------------------------------------------------------
print("Building VPP QUBO from sample_and_build()...")
H, u, tau, data = sample_and_build(seed=3)
Q, offset, C, decode = data["Q"], data["offset"], data["C"], data["decode"]

print("Channel matrix H:\n", H)
print("User symbols u:", u)
print("Tau:", tau)
print("QUBO matrix shape:", Q.shape)
print("Offset value:", offset)

n_vars = Q.shape[0]
tol = 1e-5

# --------------------------------------------------------------
# 1.5 Brute-force reference (BEFORE any annealing)
# --------------------------------------------------------------
print("\nComputing brute-force reference...")
res_brute = brute_force_check(H, u, tau, Q, offset, C, decode)

if not res_brute["ok"]:
    print("Brute-force QUBO check skipped:", res_brute["reason"])
    E_ground_raw = None
else:
    E_direct = float(res_brute["E_direct"])
    E_qubo_total = float(res_brute["E_qubo"])   # q^T Q q + offset
    l_direct = res_brute["l_direct"]
    l_from_qubo = res_brute["l_from_qubo"]

    E_ground_raw = E_qubo_total - offset

    print("Brute-force physical energy v^T G v:", E_direct)
    print("Brute-force QUBO energy q^T Q q + offset:", E_qubo_total)
    print("Raw ground-state QUBO energy:", E_ground_raw)
    print("Best integer l (direct search):", l_direct)
    print("Best integer from QUBO decode:", l_from_qubo)

# --------------------------------------------------------------
# 2. Initialize PySA solver
# --------------------------------------------------------------
solver = Solver(problem=Q, problem_type="qubo", float_type="float32")

# Common SA / PT parameters (from professor)
n_sweeps = 50
n_reads = 1000
n_replicas_pt = 2

# --------------------------------------------------------------
# 3. Helpers
# --------------------------------------------------------------
def extract_states_energies(res):
    if "energies" in res and "states" in res:
        energies = np.array([e for arr in res["energies"] for e in arr])
        states = np.array([s for arr in res["states"] for s in arr])
    elif "E" in res and "X" in res:
        energies = np.array(res["E"])
        states = np.array(res["X"])
    elif "samples" in res:
        energies = np.array(res["energies"])
        states = np.array(res["samples"])
    else:
        raise KeyError(f"Unexpected result structure: {res.keys()}")
    return states, energies


def greedy_polish_state(x, Q):
    x = np.asarray(x, dtype=int).copy()
    improved = True
    while improved:
        improved = False
        e0 = x @ Q @ x
        for i in range(len(x)):
            x[i] = 1 - x[i]
            e1 = x @ Q @ x
            if e1 < e0:
                e0 = e1
                improved = True
            else:
                x[i] = 1 - x[i]
    return x


def compute_pmf(energies, round_decimals=8):
    E_round = np.round(energies, round_decimals)
    unique_E, counts = np.unique(E_round, return_counts=True)
    probs = counts / counts.sum()
    return unique_E, probs, counts


# --------------------------------------------------------------
# 4. Temperature sweep: pure SA (single replica, T fixed)
# --------------------------------------------------------------
if E_ground_raw is not None:
    print("\n=== Simulated Annealing temperature sweep (single replica) ===")
    temp_list = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

    sa_records = []

    for T in temp_list:
        print(f"\n[SA] T = {T}")
        start = time()
        res_sa = solver.metropolis_update(
            num_sweeps=n_sweeps,
            num_reads=n_reads,
            num_replicas=1,
            update_strategy="sequential",
            min_temp=T,
            max_temp=T,
            initialize_strategy="random",
            recompute_energy=True,
            sort_output_temps=True,
            parallel=False,
            verbose=False,
        )
        elapsed = time() - start

        #corrected sweep
        states_sa, _energies_internal = extract_states_energies(res_sa)
        energies_sa = np.array([s @ Q @ s for s in states_sa], float)


        hit = np.isclose(energies_sa, E_ground_raw, atol=tol)
        success = float(np.mean(hit))
        print(f"Ground-state hit rate at T={T}: {success*100:.2f}%")
        plt.figure(figsize=(6, 4))
        plt.hist(energies_sa, bins=60, edgecolor="black", log=True)
        plt.axvline(E_ground_raw, color="red", linestyle="--")
        plt.title(f"SA Energy PMF at T={T:.5g}")
        plt.xlabel("Raw energy q^T Q q")
        plt.ylabel("Occurrences (log scale)")
        plt.tight_layout()
        plt.savefig(f"pmf_sa_T_{T:.5g}.png", dpi=200)
        plt.close()

        sa_records.append(
            {
                "temp": T,
                "success_rate": success,
                "runtime_s": elapsed,
                "best_energy_raw": float(np.min(energies_sa)),
            }
        )

    df_sa_sweep = pd.DataFrame(sa_records)
    df_sa_sweep.to_csv("sa_temp_sweep.csv", index=False)

    print("\nSA sweep summary:")
    print(df_sa_sweep)

    idx_best_sa = df_sa_sweep["success_rate"].idxmax()
    best_T_sa = float(df_sa_sweep.loc[idx_best_sa, "temp"])
    print(f"\nBest SA temperature by success_rate: {best_T_sa}")

    plt.figure(figsize=(6, 4))
    plt.plot(df_sa_sweep["temp"], df_sa_sweep["success_rate"], marker="o")
    plt.xscale("log")
    plt.xlabel("Temperature T")
    plt.ylabel("P(hit ground state)")
    plt.title("SA: ground-state probability vs temperature")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("sa_success_vs_temp.png", dpi=200)
    plt.close()
else:
    print("\nSkipping SA temperature sweep because ground state is unknown.")
    best_T_sa = None

# --------------------------------------------------------------
# 5. Parallel tempering grid sweep over (min_temp, max_temp)
# --------------------------------------------------------------
pt_records = []

if E_ground_raw is not None:
    print("\n=== Parallel tempering grid over (min_temp, max_temp) ===")

    min_temp_list = np.linspace(0.0002, 0.02, 6)
    max_temp_list = np.linspace(0.03, 0.1, 6)

    for Tmin in min_temp_list:
        for Tmax in max_temp_list:
            if Tmax <= Tmin:
                continue

            print(f"\n[PT] Tmin = {Tmin:.5g}, Tmax = {Tmax:.5g}")
            start = time()
            res_pt = solver.metropolis_update(
                num_sweeps=n_sweeps,
                num_reads=n_reads,
                num_replicas=n_replicas_pt,
                update_strategy="sequential",
                min_temp=Tmin,
                max_temp=Tmax,
                initialize_strategy="random",
                recompute_energy=True,
                sort_output_temps=True,
                parallel=False,
                verbose=False,
            )
            elapsed = time() - start
            states_pt, _energies_internal = extract_states_energies(res_pt)
            energies_pt = np.array([s @ Q @ s for s in states_pt], float)


            hit = np.isclose(energies_pt, E_ground_raw, atol=tol)
            success = float(np.mean(hit))
            print(f"Ground-state hit rate: {success*100:.2f}%")

            plt.figure(figsize=(6, 4))
            plt.hist(energies_pt, bins=60, edgecolor="black", log=True)
            plt.axvline(E_ground_raw, color="red", linestyle="--")
            plt.title(f"PT Energy PMF Tmin={Tmin:.5g}, Tmax={Tmax:.5g}")
            plt.xlabel("Raw energy q^T Q q")
            plt.ylabel("Occurrences (log scale)")
            plt.tight_layout()
            plt.savefig(
                f"pmf_pt_Tmin_{Tmin:.5g}_Tmax_{Tmax:.5g}.png",
                dpi=200
            )
            plt.close()

            pt_records.append(
                {
                    "min_temp": Tmin,
                    "max_temp": Tmax,
                    "success_rate": success,
                    "runtime_s": elapsed,
                    "best_energy_raw": float(np.min(energies_pt)),
                }
            )

    df_pt_sweep = pd.DataFrame(pt_records)
    df_pt_sweep.to_csv("pt_param_sweep.csv", index=False)

    print("\nPT sweep summary (first few rows):")
    print(df_pt_sweep.head())

    idx_best_pt = df_pt_sweep["success_rate"].idxmax()
    best_min_temp = float(df_pt_sweep.loc[idx_best_pt, "min_temp"])
    best_max_temp = float(df_pt_sweep.loc[idx_best_pt, "max_temp"])
    best_success_pt = float(df_pt_sweep.loc[idx_best_pt, "success_rate"])
    print(
        f"\nBest PT parameters by success_rate: "
        f"Tmin={best_min_temp}, Tmax={best_max_temp}, "
        f"P(hit ground)={best_success_pt*100:.2f}%"
    )

    pivot = df_pt_sweep.pivot(index="min_temp", columns="max_temp", values="success_rate")

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max(),
        ],
        interpolation="nearest",
    )
    plt.colorbar(im, label="P(hit ground state)")
    plt.xlabel("max_temp")
    plt.ylabel("min_temp")
    plt.title("PT: ground-state probability over (min_temp, max_temp)")
    plt.tight_layout()
    plt.savefig("pt_success_heatmap.png", dpi=200)
    plt.close()
else:
    print("\nSkipping PT grid sweep because ground state is unknown.")
    df_pt_sweep = None
    best_min_temp = 0.05
    best_max_temp = 0.08
    best_success_pt = 0.0

# --------------------------------------------------------------
# 6. Final PT run at best parameters and greedy polish
# --------------------------------------------------------------
print("\n=== Final PT run at best parameters ===")
print(f"Using Tmin={best_min_temp}, Tmax={best_max_temp}, replicas={n_replicas_pt}")

start = time()
res_final = solver.metropolis_update(
    num_sweeps=n_sweeps,
    num_reads=n_reads,
    num_replicas=n_replicas_pt,
    update_strategy="sequential",
    min_temp=best_min_temp,
    max_temp=best_max_temp,
    initialize_strategy="random",
    recompute_energy=True,
    sort_output_temps=True,
    parallel=False,
    verbose=True,
)
end = time()
print(f"Final PT run finished in {end - start:.2f} s")

states_final, _energies_internal = extract_states_energies(res_final)
energies_final = np.array([s @ Q @ s for s in states_final], float)

if E_ground_raw is not None:
    success_final = np.mean(np.isclose(energies_final, E_ground_raw, atol=tol))
    print(f"Final PT raw hit rate: {success_final*100:.2f}%")
else:
    success_final = None
    print("Ground state is unknown, cannot compute final hit rate.")

best_idx = np.argmin(energies_final)
best_state = states_final[best_idx]
best_energy_raw = float(energies_final[best_idx])
best_total_energy = best_energy_raw + offset

print(f"\nBest raw QUBO energy from final PT: {best_energy_raw:.8f}")
print(f"Best total physical energy:          {best_total_energy:.8f}")

l_solution = decode(best_state)
print("Decoded perturbation vector l* (from final PT):", l_solution)

polished = [greedy_polish_state(s, Q) for s in states_final]
energies_polished = np.array([p @ Q @ p for p in polished], dtype=float)

if E_ground_raw is not None:
    success_polished = np.mean(np.isclose(energies_polished, E_ground_raw, atol=tol))
    print(f"Ground-state hit rate after polish: {success_polished*100:.2f}%")
else:
    success_polished = None
    print("Ground state is unknown, cannot compute polished hit rate.")

best_pol_idx = int(np.argmin(energies_polished))
best_pol_state = polished[best_pol_idx]
best_pol_energy_raw = float(energies_polished[best_pol_idx])
best_pol_total = best_pol_energy_raw + offset

print(f"Best polished raw QUBO energy:       {best_pol_energy_raw:.8f}")
print(f"Best polished total physical energy: {best_pol_total:.8f}")
l_solution_pol = decode(best_pol_state)
print("Decoded perturbation l* after polish:", l_solution_pol)

# --------------------------------------------------------------
# 7. Save annealing results
# --------------------------------------------------------------
np.savez(
    "vpp_pysa_results.npz",
    best_state=best_state,
    best_energy_raw=best_energy_raw,
    best_total_energy=best_total_energy,
    best_pol_state=best_pol_state,
    best_pol_energy_raw=best_pol_energy_raw,
    best_pol_total=best_pol_total,
    H=H,
    u=u,
    tau=tau,
    l_solution=l_solution,
    l_solution_pol=l_solution_pol,
)

np.savetxt("vpp_pysa_energies_final.csv", np.c_[energies_final], delimiter=",", header="Energies")

with open("vpp_pysa_results.txt", "w") as f:
    if E_ground_raw is not None:
        f.write(f"SA best temperature: {best_T_sa}\n")
    if df_pt_sweep is not None:
        f.write(
            f"Best PT params: Tmin={best_min_temp}, Tmax={best_max_temp}, "
            f"P_raw={best_success_pt*100:.2f}%\n"
        )
    if success_final is not None:
        f.write(f"Final PT raw hit rate: {success_final*100:.2f}%\n")
    if success_polished is not None:
        f.write(f"Final PT polished hit rate: {success_polished*100:.2f}%\n")
    f.write(f"Best raw energy: {best_energy_raw:.8f}\n")
    f.write(f"Best total energy: {best_total_energy:.8f}\n")
    f.write(f"Best polished raw energy: {best_pol_energy_raw:.8f}\n")
    f.write(f"Best polished total energy: {best_pol_total:.8f}\n")
    f.write(f"Decoded l* (final PT): {l_solution}\n")
    f.write(f"Decoded l* (polished): {l_solution_pol}\n")

print("\nAnnealing results saved to vpp_pysa_results.*")

plt.figure(figsize=(8, 5))
plt.hist(energies_final, bins=80, edgecolor="black", log=True)
if E_ground_raw is not None:
    plt.axvline(E_ground_raw, color="red", linestyle="--", label="Ground state (raw)")
plt.xlabel("QUBO energy (raw q^T Q q)")
plt.ylabel("Occurrences (log scale)")
plt.title(f"Final PT energy distribution over {n_reads} samples")
plt.legend()
plt.tight_layout()
plt.savefig("vpp_energy_histogram_final_pt.png", dpi=200)
plt.close()

# --------------------------------------------------------------
#  3D surface plot: P(hit ground state) vs (min_temp, max_temp)
# --------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

if df_pt_sweep is not None:
    # Pivot table already used for heatmap
    pivot = df_pt_sweep.pivot(index="min_temp", columns="max_temp", values="success_rate")

    Tmin_vals = pivot.index.values
    Tmax_vals = pivot.columns.values
    Z = pivot.values

    TMAX, TMIN = np.meshgrid(Tmax_vals, Tmin_vals)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        TMIN,
        TMAX,
        Z,
        cmap="viridis",
        edgecolor="none",
        antialiased=True,
    )

    ax.set_xlabel("Low Temp")
    ax.set_ylabel("High Temp")
    ax.set_zlabel("P(hit ground)")

    ax.set_title("PT probability surface (similar to professor's plot)")
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig("pt_success_3d_surface.png", dpi=220)
    plt.close()

# --------------------------------------------------------------
# Hook for hit rate vs tau using PySA
# # --------------------------------------------------------------

def hit_rate_vs_tau_sweep(
    tau_list,
    Nr=4,
    Nt=4,
    modulation="qpsk",
    n_sweeps=50,
    n_reads=1000,
    n_replicas_pt=2,
    seed=0,
):
    records = []
    for tau in tau_list:
        rec = hit_rate_for_tau(
            tau,
            Nr=Nr,
            Nt=Nt,
            modulation=modulation,
            n_sweeps=n_sweeps,
            n_reads=n_reads,
            n_replicas_pt=n_replicas_pt,
            seed=seed,
        )
        print(
            f"tau={tau}, hit_rate={rec['hit_rate']:.3f}, E_star={rec['E_star']:.4g}"
        )
        records.append(rec)
    return records




def hit_rate_for_tau(
    tau,
    Nr=4,
    Nt=4,
    modulation="qpsk",
    n_sweeps=50,
    n_reads=1000,
    n_replicas_pt=2,
    seed=0,
):
    from numpy.random import default_rng

    rng = default_rng(seed)
    H = (rng.normal(size=(Nr, Nt)) + 1j * rng.normal(size=(Nr, Nt))) / np.sqrt(2.0)
    u = draw_symbols(Nr, modulation=modulation, rng=rng)

    # Build QUBO
    data = qubo_from_vpp(H, u, tau, t=1)
    Q = data["Q"]
    offset = data["offset"]
    C = data["C"]
    decode = data["decode"]

    # Ground truth solution via brute force
    bf = brute_force_check(H, u, tau, Q, offset, C, decode, t=1)
    if not bf["ok"]:
        raise RuntimeError(f"Brute force failed: {bf['reason']}")
    l_star = bf["l_direct"]
    E_star = bf["E_direct"]

    # -----------------------------------------------------
    #   PySA Solver — CORRECT usage
    # -----------------------------------------------------
    solver = Solver(Q, "qubo")      # constructor accepts ONLY (problem, type)

    # Now run PT-SA via metropolis_update()
    res = solver.metropolis_update(
        num_sweeps=n_sweeps,
        num_reads=n_reads,
        num_replicas=n_replicas_pt,
        min_temp=0.3,
        max_temp=1.5,
        update_strategy="random",
        initialize_strategy="random",
        parallel=True,
        use_pt=True,
        verbose=False,
    )

    # Extract states from PySA DataFrame
    states = np.array(list(res["states"]))    # shape: (num_reads, num_replicas, n_vars)
    energies = np.array(list(res["energies"]))

    hits = 0
    total = states.shape[0] * states.shape[1]

    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            q = states[i, j]
            l_hat = decode(q)
            if np.array_equal(l_hat, l_star):
                hits += 1

    hit_rate = hits / total if total > 0 else 0.0

    return {
        "tau": tau,
        "hit_rate": hit_rate,
        "E_star": float(E_star),
    }



