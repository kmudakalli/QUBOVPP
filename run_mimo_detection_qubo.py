import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

from vpp_qubo.encoding import complex_to_real
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

def qubo_from_mimo_detection(H, y, constellation="qpsk"):
    """
    ML detection (real domain) for QPSK:
        minimize || y_r - H_r x ||^2
    where x ∈ {±1}^{2Nt}  (real and imag bits).
    Convert to QUBO over q ∈ {0,1}^{2Nt} using x = 2q - 1.
    """
    H = np.asarray(H, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)

    Hr = complex_to_real(H)          # (2Nr) x (2Nt)
    yr = complex_to_real(y)          # (2Nr,)

    A = Hr.T @ Hr                    # (2Nt)x(2Nt)
    c = -2.0 * (Hr.T @ yr)           # (2Nt,)

    # Energy in x: E = x^T A x + c^T x + const
    # Substitute x = 2q - 1:
    # E = (2q-1)^T A (2q-1) + c^T (2q-1) + const
    #   = 4 q^T A q + (-4 A 1 + 2 c)^T q + const'
    n = A.shape[0]
    ones = np.ones(n)

    Q = 4.0 * A
    b = (-4.0 * (A @ ones) + 2.0 * c)

    # move linear b into diagonal: q^T Q q + sum_i b_i q_i
    Q = 0.5 * (Q + Q.T)
    for i in range(n):
        Q[i, i] += b[i]
    Q = 0.5 * (Q + Q.T)

    return Q

def main():
    Nr = Nt = 4
    rng = default_rng(0)

    H = random_channel(Nr, Nt, rng)

    # create a noiseless received vector for a random QPSK transmit vector
    s = draw_symbols(Nt, modulation="qpsk", rng=rng)      # transmitted symbols (Nt,)
    y = H @ s                                             # received (Nr,)

    Q = qubo_from_mimo_detection(H, y)

    print("\nMIMO detection (QPSK) diagonal/off-diagonal metrics:\n")
    print(diag_off_stats(Q))

    # For a single case, normalize by its own max
    global_max = np.max(np.abs(Q))

    A = np.abs(Q) / global_max
    heat(A, "QPSK MIMO detection |Q| (normalized)", "mimo_qpsk_globalnorm.png")

    B = np.abs(Q.copy())
    np.fill_diagonal(B, 0.0)
    B = B / global_max
    heat(B, "QPSK MIMO detection offdiag |Q| (normalized)", "mimo_qpsk_offdiag_globalnorm.png")

if __name__ == "__main__":
    main()