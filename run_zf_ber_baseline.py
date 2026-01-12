import numpy as np
from numpy.random import default_rng

from vpp_qubo.vpp import zf_precoder
from vpp_qubo.modulation_utils import draw_symbols, nearest_symbol

# ----------------------------
# Channel and noise utilities
# ----------------------------

def random_channel(Nr, Nt, rng):
    return (rng.normal(size=(Nr, Nt)) + 1j * rng.normal(size=(Nr, Nt))) / np.sqrt(2)

def awgn_from_ebn0(y_noiseless, ebn0_dB, bits_per_sym, rng):
    Es = np.mean(np.abs(y_noiseless)**2)
    Eb = Es / bits_per_sym
    ebn0 = 10.0**(ebn0_dB / 10.0)
    N0 = Eb / ebn0
    sigma = np.sqrt(N0 / 2.0)
    n = sigma * (
        rng.normal(size=y_noiseless.shape)
        + 1j * rng.normal(size=y_noiseless.shape)
    )
    return y_noiseless + n

# ----------------------------
# Bit utilities
# ----------------------------

def bits_qpsk(sym):
    b0 = 1 if np.real(sym) >= 0 else 0
    b1 = 1 if np.imag(sym) >= 0 else 0
    return np.array([b0, b1], dtype=int)

def bits_from_vec(u):
    return np.concatenate([bits_qpsk(s) for s in u.reshape(-1)])

# ----------------------------
# BER simulation: plain ZF
# ----------------------------

def simulate_ber_zf(
    Nr=4,
    Nt=4,
    ebn0s=(12, 13, 14, 15, 16, 17, 18),
    n_frames=1000,
    seed=0
):
    rng = default_rng(seed)

    # Fixed channel (same as VPP BER)
    H = random_channel(Nr, Nt, rng)
    P = zf_precoder(H)

    ber_list = []

    for ebn0_dB in ebn0s:
        bit_err = 0
        bit_tot = 0

        for _ in range(n_frames):
            # QPSK symbols
            u = draw_symbols(Nr, modulation="qpsk", rng=rng)

            # ZF transmit
            x_unnorm = P @ u
            Pt = float(np.vdot(x_unnorm, x_unnorm).real)
            if Pt < 1e-12:
                continue

            x = x_unnorm / np.sqrt(Pt)

            # Channel + noise
            y_noiseless = H @ x
            y = awgn_from_ebn0(
                y_noiseless,
                ebn0_dB=ebn0_dB,
                bits_per_sym=2,  # QPSK
                rng=rng
            )

            # Receiver: direct slicing
            z = y * np.sqrt(Pt)
            u_hat = nearest_symbol(z, modulation="qpsk")

            # BER counting
            b = bits_from_vec(u)
            b_hat = bits_from_vec(u_hat)
            bit_err += int(np.sum(b != b_hat))
            bit_tot += b.size

        ber = bit_err / bit_tot
        ber_list.append(ber)

    return np.array(ebn0s), np.array(ber_list)

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    ebn0s = (12, 13, 14, 15, 16, 17, 18)

    snr, ber = simulate_ber_zf(
        Nr=4,
        Nt=4,
        ebn0s=ebn0s,
        n_frames=1000,
        seed=0
    )

    print("\nZF baseline (no VPP):")
    for s, b in zip(snr, ber):
        print(f"  Eb/N0 = {s:>4.1f} dB  BER = {b:.4e}")
