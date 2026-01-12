from vpp_pysa_runner import hit_rate_vs_tau_sweep

if __name__ == "__main__":
    taus = [1.0, 2.0, 4.0, 8.0, 12.0]

    records = hit_rate_vs_tau_sweep(
        taus,
        Nr=4,
        Nt=4,
        modulation="qpsk",
        n_sweeps=50,
        n_reads=1000,
        n_replicas_pt=2,
        seed=0,
    )

    print("\nFinal results:")
    for r in records:
        print(r)
