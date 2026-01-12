import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

from vpp_pysa_runner import qubo_from_vpp
from vpp_analysis_utils import draw_symbols

def qubo_diag_offdiag_stats(Q):
    A = np.abs(Q)
    diag = np.diag(A)
    off = A.copy()
    np.fill_diagonal(off, 0.0)

    max_diag = float(diag.max())
    max_off  = float(off.max())
    mean_diag = float(diag.mean())
    mean_off  = float(off[off > 0].mean())

    return {
        "max_diag": max_diag,
        "max_offdiag": max_off,
        "ratio_max_off_to_diag": max_off / max_diag if max_diag > 0 else float("inf"),
        "ratio_mean_off_to_diag": mean_off / mean_diag if mean_diag > 0 else float("inf"),
    }


def random_channel(Nr, Nt, rng):
    return (rng.normal(size=(Nr, Nt)) + 1j * rng.normal(size=(Nr, Nt))) / np.sqrt(2)

def metrics(Q):
    Q = np.asarray(Q, dtype=float)
    d = np.abs(np.diag(Q))
    off = np.abs(Q.copy())
    np.fill_diagonal(off, 0.0)
    max_d = float(np.max(d))
    max_off = float(np.max(off))
    mean_d = float(np.mean(d))
    mean_off = float(np.mean(off))
    return {
        "max_diag": max_d,
        "max_off": max_off,
        "ratio_max_off_to_diag": (max_off / max_d) if max_d > 0 else np.inf,
        "ratio_mean_off_to_diag": (mean_off / mean_d) if mean_d > 0 else np.inf,
    }

def heat(A, title, fname):
    plt.figure(figsize=(5,4))
    im = plt.imshow(A, interpolation="nearest", aspect="auto")
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

def main():
    Nr = Nt = 4
    taus = [1.0, 2.0, 4.0, 8.0, 12.0]
    rng = default_rng(0)

    H = random_channel(Nr, Nt, rng)
    u = draw_symbols(Nr, modulation="qpsk", rng=rng)

    Qs = {}
    for tau in taus:
        Q = qubo_from_vpp(H, u, tau, t=1)["Q"]
        Qs[tau] = Q
        # ---- ADD THIS ONLY FOR tau = 4 (conventional VPP) ----
        if tau == 4.0:
            print("\nNumerical QUBO entries for tau = 4 (QPSK VPP)\n")

            # Show a small concrete block
            print("Top-left 6x6 block of Q:")
            print(np.array2string(Q[:6, :6], precision=3, suppress_small=True))

            # Show largest off-diagonal coefficients
            A = np.abs(Q)
            np.fill_diagonal(A, 0.0)

            idx = np.dstack(
                np.unravel_index(np.argsort(A.ravel())[::-1], A.shape)
            )[0]

            print("\nTop 10 off-diagonal |Q_ij| entries:")
            count = 0
            for i, j in idx:
                if i == j:
                    continue
                print(f"  Q[{i},{j}] = {Q[i,j]:.6g}   |Q| = {abs(Q[i,j]):.6g}")
                count += 1
                if count == 10:
                    break
    # Global max across all taus
    global_max = max(np.max(np.abs(Q)) for Q in Qs.values())

    print("\nDiagonal/off-diagonal metrics (absolute, not normalized):\n")
    for tau in taus:
        m = metrics(Qs[tau])
        print(f"tau={tau:>4}: {m}")

    # Heatmaps with global normalization
    for tau in taus:
        A = np.abs(Qs[tau]) / global_max
        heat(A, f"|Q| / global max, tau={tau}", f"globalnorm_tau_{tau}.png")

    # Off-diagonal only heatmaps with global normalization
    for tau in taus:
        A = np.abs(Qs[tau].copy())
        np.fill_diagonal(A, 0.0)
        A = A / global_max
        heat(A, f"Off-diagonal |Q| / global max, tau={tau}", f"offdiag_globalnorm_tau_{tau}.png")



if __name__ == "__main__":
    main()