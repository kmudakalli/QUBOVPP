#modulation helpers and conventional tau
import numpy as np

# ---------------------------
# Modulation utilities
# ---------------------------

def qpsk_constellation():
    # Unnormalized QPSK: real and imag in { -1, +1 }
    pts = np.array(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
        dtype=np.complex128,
    )
    return pts

def qam16_constellation():
    # Unnormalized 16QAM: real and imag in { -3, -1, +1, +3 }
    re = np.array([-3, -1, 1, 3], dtype=np.float64)
    im = np.array([-3, -1, 1, 3], dtype=np.float64)
    pts = [x + 1j * y for x in re for y in im]
    return np.array(pts, dtype=np.complex128)

def draw_symbols(Nr, modulation="qpsk", rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if modulation.lower() == "qpsk":
        const = qpsk_constellation()
    elif modulation.lower() == "16qam":
        const = qam16_constellation()
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")
    idx = rng.integers(0, len(const), size=Nr)
    return const[idx]

# ---------------------------
# "Conventional" tau helper
# ---------------------------

def conventional_tau(modulation="qpsk"):
    """
    Conventional VP style tau from constellation geometry:

        tau = 2 * (c_max + Δ/2)

    where c_max is max |Re(symbol)| and Δ is min distance between
    adjacent points on the real axis.

    This matches tau = 4 for QPSK, tau = 8 for unnormalized 16QAM.
    """
    if modulation.lower() == "qpsk":
        re_levels = np.array([-1, 1], dtype=np.float64)
    elif modulation.lower() == "16qam":
        re_levels = np.array([-3, -1, 1, 3], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")

    re_levels = np.sort(re_levels)
    c_max = np.max(np.abs(re_levels))
    # minimum spacing on real axis
    deltas = np.diff(re_levels)
    delta_min = np.min(np.abs(deltas))
    tau = 2.0 * (c_max + 0.5 * delta_min)
    return tau
