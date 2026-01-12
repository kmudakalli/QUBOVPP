import numpy as np
from numpy.random import default_rng

from vpp_qubo.vpp import zf_precoder
from vpp_qubo.modulation_utils import draw_symbols, nearest_symbol, gamma_tau, conventional_tau

def random_channel(Nr, Nt, rng):
    return (rng.normal(size=(Nr, Nt)) + 1j*rng.normal(size=(Nr, Nt))) / np.sqrt(2)

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

def bits_qpsk(sym):
    b0 = 1 if np.real(sym) >= 0 else 0
    b1 = 1 if np.imag(sym) >= 0 else 0
    return np.array([b0,b1], dtype=int)

def bits_from_vec(u):
    return np.concatenate([bits_qpsk(s) for s in u.reshape(-1)])

def brute_force_l_min_power(H, u, tau):
    # conventional VP baseline for 4x4 using bounded integer search in {-1,0,1}^{2Nr}
    Nr = H.shape[0]
    P = zf_precoder(H)
    vals = [-1,0,1]
    cand = np.array(np.meshgrid(*([vals]*(2*Nr)), indexing="ij")).reshape(2*Nr,-1).T

    best = None
    bestPt = np.inf

    for l in cand:
        l_c = l[:Nr] + 1j*l[Nr:]
        d = u + tau*l_c
        x_unnorm = P @ d
        Pt = float(np.vdot(x_unnorm, x_unnorm).real)
        if Pt < bestPt:
            bestPt = Pt
            best = l_c

    return best, bestPt

def simulate_ber(Nr=4, Nt=4, taus=(1,2,4,8,12), snrs=(12,13,14,15,16,17,18), n_frames=500, seed=0):
    rng = default_rng(seed)
    H = random_channel(Nr, Nt, rng)
    P = zf_precoder(H)

    results = {}

    for tau in taus:
        ber_snr = []
        for snr_dB in snrs:
            bit_err = 0
            bit_tot = 0

            for _ in range(n_frames):
                u = draw_symbols(Nr, modulation="qpsk", rng=rng)

                l_star, Pt = brute_force_l_min_power(H, u, tau)
                d = u + tau*l_star

                x = (P @ d) / np.sqrt(Pt)             # power normalized transmit
                y_noiseless = H @ x                   # = d / sqrt(Pt)
                y = awgn_from_ebn0(
                    y_noiseless,
                    ebn0_dB=snr_dB,
                    bits_per_sym=2,  # QPSK
                    rng=rng
                )

                # receiver: undo scalar normalization then modulo then slice
                z = y * np.sqrt(Pt)                   # = d + noise'
                z_mod = gamma_tau(z, tau)             # modulo-tau
                u_hat = nearest_symbol(z_mod, modulation="qpsk")

                b = bits_from_vec(u)
                b_hat = bits_from_vec(u_hat)
                bit_err += int(np.sum(b != b_hat))
                bit_tot += b.size

            ber = bit_err / bit_tot
            ber_snr.append(ber)

        results[tau] = (np.array(snrs), np.array(ber_snr))

    return results

if __name__ == "__main__":
    taus = [1.0, 2.0, 4.0, 8.0, 12.0]
    print("Conventional tau for QPSK =", conventional_tau("qpsk"))

    res = simulate_ber(
        Nr=4, Nt=4,
        taus=taus,
        snrs=(12, 13, 14, 15, 16, 17, 18),
        n_frames=500,
        seed=0
    )

    for tau in taus:
        snr, ber = res[tau]
        print(f"\nTau={tau}")
        for s,b in zip(snr, ber):
            print(f"  SNR={s:>4.1f} dB  BER={b:.4e}")
