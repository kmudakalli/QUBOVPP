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
def build_integer_encoding(n_vars):
    """
    Binary encoding for l in {-1, 0, 1}^n_vars.
    Each integer uses 2 bits with weights [-1, 1]:
        (q0, q1) ∈ {0,1}^2
        l = -q0 + q1 ∈ {-1, 0, 1}
    """
    C = np.zeros((n_vars, 2 * n_vars), dtype=int)
    weights = np.array([-1, 1], dtype=int)
    for k in range(n_vars):
        C[k, 2*k:2*k+2] = weights

    def decode(q):
        q = np.asarray(q).reshape(-1)
        v = C @ q
        return v.astype(int)

    return C, decode
