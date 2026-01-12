'''
Gives BER vs SNR curves for several tau vals
for small systems, exact or near exact VP via brute force l search
'''

import numpy as np
from numpy.random import default_rng

from vpp_qubo.vpp import (
    zf_precoder,
    gram_of_map,
    build_basis_map,
)
from vpp_qubo.encoding import complex_to_real
from vpp_analysis_utils import (
    draw_symbols,
    qpsk_constellation,
    qam16_constellation,
    conventional_tau,
)

rng = default_rng()


def bits_qpsk(x):
    """
    Simple Gray-ish mapping for QPSK:
      Re < 0 -> 0, Re >= 0 -> 1
      Im < 0 -> 0, Im >= 0 -> 1
    Returns shape (2,) bits for a single complex symbol.
    """
    b0 = 1 if np.real(x) >= 0 else 0
    b1 = 1 if np.imag(x) >= 0 else 0
    return np.array([b0, b1], dtype=int)


def bits_16qam(x):
    """
    Gray-coded bits for unnormalized 16QAM with re, im in {-3, -1, 1, 3}.
    Mapping (from low to high amplitude):
      -3 -> 00
      -1 -> 01
       1 -> 11
       3 -> 10
    Compose for real and imag to get 4 bits per symbol.
    """
    def axis_bits(v):
        if v < -2:
            return np.array([0, 0], dtype=int)
        elif v < 0:
            return np.array([0, 1], dtype=int)
        elif v < 2:
            return np.array([1, 1], dtype=int)
        else:
            return np.array([1, 0], dtype=int)

    re = np.real(x)
    im = np.imag(x)
    return np.concatenate([axis_bits(re), axis_bits(im)])


def bits_from_symbols(vec, modulation="qpsk"):
    bits_list = []
    for s in vec:
        if modulation.lower() == "qpsk":
            bits_list.append(bits_qpsk(s))
        elif modulation.lower() == "16qam":
            bits_list.append(bits_16qam(s))
        else:
            raise ValueError(f"Unsupported modulation: {modulation}")
    return np.concatenate(bits_list)


def quantize_to_constellation(z, modulation="qpsk"):
    """
    Nearest point on the constellation.
    """
    if modulation.lower() == "qpsk":
        const = qpsk_constellation()
    elif modulation.lower() == "16qam":
        const = qam16_constellation()
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")
    z = np.asarray(z, dtype=np.complex128)
    out = np.zeros_like(z)
    for i, val in enumerate(z):
        idx = np.argmin(np.abs(val - const))
        out[i] = const[idx]
    return out


def gamma_tau(x, tau):
    """
    Gamma_tau as in the LRA QVP paper:
      Theta_tau(x) = tau * floor(x / tau + 1/2)
      Gamma_tau(x) = x - Theta_tau(x)
    Applied elementwise to a complex vector.
    """
    x = np.asarray(x, dtype=np.complex128)
    re = np.real(x)
    im = np.imag(x)

    def gamma_real(r):
        return r - tau * np.floor(r / tau + 0.5)

    return gamma_real(re) + 1j * gamma_real(im)


def vpp_precoder_conventional(H, u, tau, search_L=1):
    """
    Conventional VP baseline:
    - ZF precoder P = H^H (HH^H)^-1
    - Real lattice basis F from P
    - Gram matrix G = F^T F
    - Brute force search over integer vector l in [-L..L]^(2Nr)
    for the perturbation that minimizes transmit power.

    Returns:
      x: transmit vector (Nt,)
      l_complex: optimal perturbation vector in C^Nr
    """
    H = np.asarray(H, dtype=np.complex128)
    u = np.asarray(u, dtype=np.complex128).reshape(-1)
    Nr, Nt = H.shape

    P = zf_precoder(H)           # Nt x Nr
    F = build_basis_map(P)       # 2*Nt x 2*Nr
    G = gram_of_map(F)           # 2*Nr x 2*Nr

    y = complex_to_real(u).reshape(-1)

    vals = np.arange(-search_L, search_L + 1, dtype=int)
    cand = np.array(
        np.meshgrid(*([vals] * (2 * Nr)), indexing="ij")
    ).reshape(2 * Nr, -1).T

    best_E = np.inf
    best_l_real = None

    for l in cand:
        v = y + tau * l
        E = float(v @ G @ v)
        if E < best_E:
            best_E = E
            best_l_real = l.copy()

    # Map real vector [Re(l); Im(l)] to complex l
    l_real = best_l_real[:Nr]
    l_imag = best_l_real[Nr:]
    l_complex = l_real + 1j * l_imag

    x = P @ (u + tau * l_complex)
    return x, l_complex


def awgn(y, snr_dB, rng=None):
    """
    Add AWGN so that SNR per symbol equals snr_dB,
    assuming unit average symbol energy.
    """
    if rng is None:
        rng = default_rng()
    y = np.asarray(y, dtype=np.complex128)
    Es = np.mean(np.abs(y) ** 2)
    snr_linear = 10.0 ** (snr_dB / 10.0)
    N0 = Es / snr_linear
    sigma = np.sqrt(N0 / 2.0)
    noise = sigma * (rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape))
    return y + noise


def simulate_vpp_ber(
    Nr=4,
    Nt=4,
    modulation="qpsk",
    tau_list=(1.0, 4.0, 12.0),
    snr_list_dB=(0.0, 5.0, 10.0, 15.0, 20.0),
    n_frames=2000,
    search_L=1,
    seed=0,
):
    """
    Monte Carlo BER for conventional VP baseline, for different tau values.
    Uses one random H per run for simplicity, but you can extend to H ensemble.
    """

    rng = default_rng(seed)
    H = (rng.normal(size=(Nr, Nt)) + 1j * rng.normal(size=(Nr, Nt))) / np.sqrt(2.0)

    print(f"\n[Conventional VPP BER] Nr={Nr}, Nt={Nt}, modulation={modulation}")
    print(f"search_L={search_L}, one random H per tau\n")

    results = {}

    for tau in tau_list:
        ber_per_snr = []
        print(f"tau = {tau}")

        for snr_dB in snr_list_dB:
            bit_err = 0
            bit_total = 0

            for _ in range(n_frames):
                u = draw_symbols(Nr, modulation=modulation, rng=rng)
                x, l = vpp_precoder_conventional(
                    H, u, tau=tau, search_L=search_L
                )

                # Downlink: y_rx = H x + n
                y_rx = awgn(H @ x, snr_dB, rng=rng)

                # With perfect ZF and no noise: y_rx = u + tau l
                # We undo modulo tau, then quantize back to constellation.
                y_mod = gamma_tau(y_rx, tau)
                u_hat = quantize_to_constellation(y_mod, modulation=modulation)

                b_true = bits_from_symbols(u, modulation=modulation)
                b_hat = bits_from_symbols(u_hat, modulation=modulation)

                bit_err += np.sum(b_true != b_hat)
                bit_total += b_true.size

            ber = bit_err / bit_total if bit_total > 0 else np.nan
            ber_per_snr.append(ber)
            print(f"  SNR={snr_dB:4.1f} dB, BER={ber:.3e}")

        results[tau] = {
            "snr_dB": np.array(snr_list_dB),
            "ber": np.array(ber_per_snr),
        }
        print("")

    return results


if __name__ == "__main__":
    # Example: QPSK 4x4, tau sweep
    taus_qpsk = [1.0, 2.0, conventional_tau("qpsk"), 6.0]
    simulate_vpp_ber(
        Nr=4,
        Nt=4,
        modulation="qpsk",
        tau_list=taus_qpsk,
        snr_list_dB=(0.0, 5.0, 10.0, 15.0, 20.0),
        n_frames=1000,
        search_L=1,
        seed=42,
    )

    # Example: 16QAM 4x4, tau sweep
    taus_16qam = [4.0, conventional_tau("16qam"), 10.0, 12.0]
    simulate_vpp_ber(
        Nr=4,
        Nt=4,
        modulation="16qam",
        tau_list=taus_16qam,
        snr_list_dB=(10.0, 15.0, 20.0, 25.0),
        n_frames=1000,
        search_L=1,
        seed=43,
    )
