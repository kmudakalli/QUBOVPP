import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

from vpp_qubo.vpp import qubo_from_vpp
from vpp_qubo.modulation_utils import draw_symbols

def random_channel(Nr, Nt, rng):
    return (rng.normal(size=(Nr, Nt)) + 1j*rng.normal(size=(Nr, Nt))) / np.sqrt(2)

def diag_off_stats(Q):
    A = np.abs(Q)
    diag = np.diag(A)
    off = A.copy()
    np.fill_diagonal(off, 0.0)
    return {
        "max_diag": float(diag.max()),
        "max_off": float(off.max()),
        "ratio_max_off_to_diag": float(off.max()/diag.max()) if diag.max() > 0 else np.inf,
        "ratio_mean_off_to_diag": float(off[off > 0].mean()/diag.mean()) if diag.mean() > 0 else np.inf,
    }

def heat(A, title, fname):
    plt.figure(figsize=(5,4))
    im = plt.imshow(A, interpolation="nearest", aspect="auto")
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("j"); plt.ylabel("i")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

def main():
    Nr = Nt = 4
    taus = [4.0, 8.0, 12.0]   # include conventional 16QAM tau=8
    rng = default_rng(0)

    H = random_channel(Nr, Nt, rng)
    u = draw_symbols(Nr, modulation="16qam", rng=rng)

    Qs = {}
    for tau in taus:
        Qs[tau] = qubo_from_vpp(H, u, tau, t=1)["Q"]

    global_max = max(np.max(np.abs(Q)) for Q in Qs.values())

    print("\n16QAM VPP diagonal/off-diagonal metrics:\n")
    for tau in taus:
        print(f"tau={tau}: {diag_off_stats(Qs[tau])}")

    for tau in taus:
        A = np.abs(Qs[tau]) / global_max
        heat(A, f"16QAM VPP |Q| / global max, tau={tau}", f"vpp16qam_globalnorm_tau_{tau}.png")

        B = np.abs(Qs[tau].copy())
        np.fill_diagonal(B, 0.0)
        B = B / global_max
        heat(B, f"16QAM VPP offdiag |Q| / global max, tau={tau}", f"vpp16qam_offdiag_globalnorm_tau_{tau}.png")

if __name__ == "__main__":
    main()