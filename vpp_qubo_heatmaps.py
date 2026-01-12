'''
Fixes one random H and u per modulation, builds Q for each tau, saves heatmaps, prints diagonal vs off diagonal statistics
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

from vpp_qubo.vpp import qubo_from_vpp
from vpp_analysis_utils import draw_symbols

rng = default_rng()


def random_channel(Nr, Nt, rng=None):
    """
    IID Rayleigh fading: complex Gaussian CN(0, 1) entries.
    """
    if rng is None:
        rng = default_rng()
    H = (rng.normal(size=(Nr, Nt)) + 1j * rng.normal(size=(Nr, Nt))) / np.sqrt(2.0)
    return H


def compute_qubo_metrics(Q):
    """
    Given symmetric Q, compute simple diagonal / off diagonal metrics.
    """
    Q = np.asarray(Q, dtype=float)
    diag = np.diag(Q)
    mask_off = ~np.eye(Q.shape[0], dtype=bool)
    off = Q[mask_off]

    max_diag = float(np.max(np.abs(diag)))
    mean_diag = float(np.mean(np.abs(diag)))
    max_off = float(np.max(np.abs(off))) if off.size > 0 else 0.0
    mean_off = float(np.mean(np.abs(off))) if off.size > 0 else 0.0
    ratio_max = max_off / max_diag if max_diag > 0 else np.inf

    return {
        "max_diag": max_diag,
        "mean_diag": mean_diag,
        "max_off": max_off,
        "mean_off": mean_off,
        "ratio_max_off_to_diag": ratio_max,
    }


def plot_qubo_heatmap(Q, title, fname):
    """
    Plot |Q| normalized by its maximum entry.
    """
    Q = np.asarray(Q, dtype=float)
    A = np.abs(Q)
    max_val = np.max(A)
    if max_val > 0:
        A = A / max_val

    plt.figure(figsize=(5, 4))
    im = plt.imshow(A, origin="upper", interpolation="nearest", aspect="auto")
    plt.colorbar(im, label="|Q_ij| / max |Q|")
    plt.title(title)
    plt.xlabel("j (binary variable index)")
    plt.ylabel("i (binary variable index)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def qubo_heatmap_experiment(
    Nr=4,
    Nt=4,
    modulation="qpsk",
    tau_list=(1.0, 4.0, 12.0),
    seed=0,
    out_prefix="qpsk_vpp",
):
    """
    For a single random H and u, build QUBO for each tau and
    save heatmaps plus print diagonal vs off diagonal metrics.
    """
    rng = default_rng(seed)
    H = random_channel(Nr, Nt, rng)
    u = draw_symbols(Nr, modulation=modulation, rng=rng)

    print(f"\n[QUBO heatmaps] modulation={modulation}, Nr={Nr}, Nt={Nt}")
    print("Random H and u fixed for all tau values\n")

    for tau in tau_list:
        data = qubo_from_vpp(H, u, tau, t=1)
        Q = data["Q"]
        metrics = compute_qubo_metrics(Q)

        print(f"tau = {tau}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4g}")
        print("")

        title = f"{modulation.upper()} VPP, tau={tau}, Nr={Nr}, Nt={Nt}"
        fname = f"{out_prefix}_tau{tau:.2f}.png".replace(".", "p")
        plot_qubo_heatmap(Q, title=title, fname=fname)


if __name__ == "__main__":
    # Example runs:
    qubo_heatmap_experiment(
        Nr=4,
        Nt=4,
        modulation="qpsk",
        tau_list=(1.0, 4.0, 12.0),
        seed=123,
        out_prefix="qpsk_vpp",
    )

    qubo_heatmap_experiment(
        Nr=4,
        Nt=4,
        modulation="16qam",
        tau_list=(4.0, 8.0, 12.0),
        seed=123,
        out_prefix="qam16_vpp",
    )
