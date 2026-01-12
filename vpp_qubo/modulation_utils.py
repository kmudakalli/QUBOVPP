import numpy as np

def qpsk_constellation():
    return np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex128)

def qam16_constellation():
    re = np.array([-3, -1, 1, 3], dtype=np.float64)
    im = np.array([-3, -1, 1, 3], dtype=np.float64)
    return np.array([x + 1j*y for x in re for y in im], dtype=np.complex128)

def draw_symbols(N, modulation="qpsk", rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if modulation.lower() == "qpsk":
        const = qpsk_constellation()
    elif modulation.lower() == "16qam":
        const = qam16_constellation()
    else:
        raise ValueError("Unsupported modulation")
    idx = rng.integers(0, len(const), size=N)
    return const[idx]

def nearest_symbol(z, modulation="qpsk"):
    if modulation.lower() == "qpsk":
        const = qpsk_constellation()
    elif modulation.lower() == "16qam":
        const = qam16_constellation()
    else:
        raise ValueError("Unsupported modulation")
    z = np.asarray(z, dtype=np.complex128).reshape(-1)
    out = np.zeros_like(z)
    for i, v in enumerate(z):
        out[i] = const[np.argmin(np.abs(v - const))]
    return out

def gamma_tau(x, tau):
    x = np.asarray(x, dtype=np.complex128)
    re = np.real(x)
    im = np.imag(x)
    def gamma(r):
        return r - tau * np.floor(r / tau + 0.5)
    return gamma(re) + 1j * gamma(im)

def conventional_tau(modulation="qpsk"):
    # tau = 2 (c_max + Delta/2) = 2*c_max + Delta (per real dimension)
    if modulation.lower() == "qpsk":
        levels = np.array([-1, 1], dtype=np.float64)
    elif modulation.lower() == "16qam":
        levels = np.array([-3, -1, 1, 3], dtype=np.float64)
    else:
        raise ValueError("Unsupported modulation")
    levels = np.sort(levels)
    cmax = float(np.max(np.abs(levels)))
    Delta = float(np.min(np.abs(np.diff(levels))))
    return 2.0 * (cmax + 0.5 * Delta)
