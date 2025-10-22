import numpy as np

def complex_to_real(M):
    """
    Maps a complex vector or matrix to its real-valued representation:
        For vector u = a + jb:  phi(u) = [Re(u); Im(u)]
        For matrix P = A + jB:  Phi(P) = [[A, -B], [B, A]]
    """

    M =np.asarray(M)

    #vector
    if M.ndim == 1:
        #complex vector u = a + jb
        ## map to real vector: [Re(u); Im(u)]
        a =M.real
        b = M.imag
        return np.concatenate([a, b], axis=0).astype(np.float64)
    
    #matrix
    elif M.ndim == 2:
        A= M.real
        B =M.imag
        top = np.concatenate([A, -B], axis=1)
        bot= np.concatenate([B,  A], axis=1)
        return np.concatenate([top, bot], axis=0).astype(np.float64)
    else:
        raise ValueError("M must be 1D or 2D")

#builds binary encoding matrix C for n integers
def build_integer_encoding(n_vars, t=1):
    cols_per =t+1  # bits per integer
    C= np.zeros((n_vars, n_vars * cols_per), dtype=int)
    
    
    # 2's complement weights: [ -2^t, 1, 2, ..., 2^(t-1) ]
    weights= np.array([-(2**t)] + [2**i for i in range(t)], dtype=int)

    # Fill block diagonal where each integer variable gets its own set of weights
    for k in range(n_vars):
        C[k, k*cols_per:(k+1)*cols_per] = weights

    def decode(q):
        q =np.asarray(q).reshape(-1)
        return (C@ q).astype(int)

    return C, decode