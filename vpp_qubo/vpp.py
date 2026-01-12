import numpy as np
from itertools import product
from .encoding import complex_to_real, build_integer_encoding

# 2102.12540v1.pdf (Downlink VPP basics): (Section 2 page 5: P  = H^H * (H*H^H)^-1 (zero forcing precoder)
def zf_precoder(H):
    H = np.asarray(H, dtype=np.complex128)
    return H.conj().T @ np.linalg.pinv(H @ H.conj().T)

'''
standard Gram matrix G = F^T * F
F is real valued basis matrix from precoder P
G encodes inner products of basis vectors

Need G to show how perturbations v affect transmit power and cost becomes
simple quadratic function of v which is needed to build QUBO
'''
def gram_of_map(F):
    return F.T @ F

'''
converts complex precoder P to lattice map so that optimization is in set of real numbers and not complex numbers
Why:
optimizatoin should be expressed over real numbers

Reference: Section 2.1: Primer Maximum Likelihood Chance paragraph 2
'''
def build_basis_map(P):
    return complex_to_real(P)


#Step 5 - encode integers w binary values
# Creates matrix C used in v =Cq where v is integer perturbation vector and q is binary vector

# n = num of integers needed to encode and t is num of magnitude bits

#FOLLOWING BELOW CONFLICTS WITH encoding.py
# def build_integer_encoding(n, t=1):

#     bits_per_int= t + 1 #t+1 for 2s complement


#     m = n *bits_per_int  #total num of binary variables is m = n * (t+1)

#     #mathematically q is element of {0,1}^m

#     #weight vector 2's complement
#     #ex. if t = 2 then weights = [-4,1,2]
#     weights = np.array([-(2**t)] + [2**i for i in range(t)], dtype=int)

#     # Build block diagonal encoding matrix
#     C = np.zeros((n, m), dtype=int)

#     #v=Cq (builds block-diagonal encoding matrix C by placing 1 copy of 
#     # weight vector in corresponding block of columns for each integer v_i)
#     for i in range(n):
#         C[i, i*bits_per_int:(i+1)*bits_per_int] = weights

#     # helper function to decode binary vector q back into integers v
#     def decode(q):
#         q = np.asarray(q).reshape(-1)
#         l = C @ q
#         return l.astype(int)

#     return C, decode


# cost function -> QUBO form
def form_qubo_core(G, y_vec, tau, C, const_offset=0.0):
    # y should be col vector
    y = y_vec.reshape(-1, 1)
    # QUADRATIC TERM
    A= tau**2 * (C.T @ G @ C)
    # LINEAR TERM
    b = 2.0 * tau * (y.T @ G @ C).reshape(-1)
    # CONSTANT TERM
    const = float((y.T @ G @ y)) + float(const_offset)
    #symmetric quadratic form 
    Q = 0.5 * (A + A.T)
    #(q_i)^2 = q_i for binary variables 
    for i in range(Q.shape[0]):
        Q[i, i]+= b[i]
    #make final Q symmetric
    Q = 0.5 * (Q + Q.T)
    #returns QUBO matrix Q and const offset
    return Q, const


def qubo_from_vpp(H, u, tau, t=1):
    # step 1 - channnel matrix H and user symbols u (complex form)
    H =np.asarray(H, dtype=np.complex128)
    u= np.asarray(u, dtype=np.complex128).reshape(-1)


    #Nr = num of users ()
    Nr = H.shape[0]

    # ZF precoding
    P = zf_precoder(H)

    # STEP 4
    # realify precoder into lattice basis F and user symbols u_r
    F= build_basis_map(P)
    u_r =complex_to_real(u)

    C, decode = build_integer_encoding(2*Nr) #removed t parameter


    # STEP 4 - GRAM MATRIX
    G = gram_of_map(F)

    #builds QUBO
    Q, const = form_qubo_core(G, u_r,tau, C, const_offset=0.0)
    return {"Q": Q, "offset": const, "C": C, "decode": decode}

#works for small cases
def brute_force_check(H, u, tau, Q, offset, C, decode, t=1):
    Nr = H.shape[0]

    F= build_basis_map(zf_precoder(H))
    G= gram_of_map(F)
    y =complex_to_real(u)


    #UPDATE from author: vals = [-1, 0, 1] as Re/Im(v = {-1, 0, 1})
    # for each component v_i = a_i = jb_i, a_i (real part) can be -1, 0, or 1, and b_i (imaginary part) can be -1, 0, or 1
    # So for each complex component v_i can take 3x3 = 9 possible complex values

    vals= [-1, 0, 1]
    cand= np.array(list(product(*([vals] * (2*Nr)))), dtype=int)


    bestE =np.inf
    bestl= None

    # Brute force paper method 
    # takes each integer vector l which are bounded by encoding.py and finds verturbation with v = y + τl and plugs into original energy
    for l in cand:
        v =y.reshape(-1) + tau * l
        E = float(v @ G @ v)
        if E< bestE:
            bestE= E
            bestl= l.copy()

    # Brute-force QUBO (to verify paper method)
    M =Q.shape[0]

    # picked 26 randomly. 2^26 is 67108864 possibile binary vectors which can take very long time to compile
    if M >26:
        return {"ok": False, "reason": "too many variables", "bestE_direct": bestE, "best_l_direct": bestl}

    bestQ= np.inf
    bestq =None

    # checks EVERY POSSIBLE BINARY VECTOR q in {0,1}^M and for each q finds QUBO energy and picks one w smallest energy
    for z in range(1 << M):
        q =np.array([(z >> i) & 1 for i in range(M)], float)
        E =float(q @ Q @ q + offset)
        if E < bestQ:
            bestQ= E
            bestq =q

    l_from_qubo = decode(bestq)
    return {"ok": True, "E_direct": bestE, "E_qubo": bestQ, "l_from_qubo": l_from_qubo, "l_direct": bestl}

#random example
def sample_and_build(seed=0):
    #initializes random num generator
    rng= np.random.default_rng(seed)
    Nr= 4
    Nt = 4

    '''
    random 2x2 channel matrix H - 2102.1540v1.pdf has channel matrix H defined as complex matrix and
    is treated as random w complex normal w zero mean (i.i.d Gaussian entries)
    used std assumption that it corresponds to Rayleigh fading channel model

    so i modeled each entry of H as a complex Gaussian random variable where each channel coefficient
    is a random complex number whose real and imaginary parts are independent Gaussians with 
    mean 0 and variance 0.5 which makes sure total avg power of each channel link is 1
    '''
    H = (rng.normal(size=(Nr, Nt)) + 1j * rng.normal(size=(Nr, Nt))) / np.sqrt(2)
    # real Gaussian random numbers w mean 0 + variance 1 + same thing but imaginary
    # divided by sqrt(2) so that complex random variable has unit variance

    #referenced "Leverage Quantum Annealing..." QPSK MODULATION
    # where QPSK each sender transmits one offour possible symbols v_i in {+/- 1 +/- 1j} 
    #QPSK symbols
    const = np.array([1+1j, 1-1j, -1+1j, -1-1j])

    #selects N_r elements from set uniformly at random which results in random QPSK symbols in length N_r vector
    u =rng.choice(const, size=Nr)

    # arbitrary spacing tau = 4.0 chosen for QPSK {+/- 1 +/- 1j} 
    tau= 4.0

    # t is num of magnitude bits in the binary encoding of each integer component of perturbation vector
    # when t = 0 each variable v_i was represented by 1 bit which acted as a binary variable
    # when t =1 each v_i now uses 2 bits (one sign bit and one magnitude bit)

    # I HAVE N_r = 2 users
    # After realification, v is in Z^(2N_r) = Z^4
    #Each integer in v now uses t + 1 = 2 bits
    # SO TOTAL BINARY VARIABLES = M  = 2*4 = 8

    #WHY THE QUBO MATRIX Q IS 8x8 instead of 2x2

    data = qubo_from_vpp(H, u, tau, t=1)
    return H, u,tau, data