import numpy as np
from numba import njit
from scipy.stats import entropy
from numpy.linalg import norm
import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), 'bin/Cfunc.so')
Cfunc = ctypes.CDLL(lib_path)

# Define the input type and return type of the C functions
Cfunc.jsd.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
Cfunc.jsd.restype = ctypes.c_double
Cfunc.avg_jsd.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                           ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                           ctypes.c_size_t, ctypes.c_size_t]
Cfunc.avg_jsd.restype = ctypes.c_double

def theta(seq1_list, seq2_list):
    list_length = len(seq1_list)
    keys = set()
    for i in range(list_length):
        keys = keys.union(seq1_list[i].keys()).union(seq2_list[i].keys())

    V1_list = []
    V2_list = []
    for i in range(list_length):
        V1 = np.array([seq1_list[i].get(k, 0.0) for k in keys], dtype=np.float64)
        V2 = np.array([seq2_list[i].get(k, 0.0) for k in keys], dtype=np.float64)
        V1_list.append((ctypes.c_double * len(V1))(*V1))
        V2_list.append((ctypes.c_double * len(V2))(*V2))

    V1_list_c = (ctypes.POINTER(ctypes.c_double) * list_length)()
    V2_list_c = (ctypes.POINTER(ctypes.c_double) * list_length)()

    for i in range(list_length):
        V1_list_c[i] = V1_list[i]
        V2_list_c[i] = V2_list[i]

    # Call the C function
    result = Cfunc.avg_jsd(V1_list_c, V2_list_c, list_length, len(V1))

    return result


# Define the input type and return type of the C functions
Cfunc.fill_matrix.argtypes = [ctypes.POINTER(ctypes.c_double),
                              ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                              ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                              ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
Cfunc.fill_matrix.restype = None



def theta_matrix(seq1_list, seq2_list):
    num_seqs1 = len(seq1_list)
    num_seqs2 = len(seq2_list)

    # Create the matrix
    matrix = np.empty((num_seqs1, num_seqs2), dtype=np.float64)
    matrix_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    keys = set()
    for seq1, seq2 in zip(seq1_list, seq2_list):
        for d1, d2 in zip(seq1, seq2):
            keys = keys.union(d1.keys()).union(d2.keys())

    V1_list = []
    V2_list = []
    for seq1, seq2 in zip(seq1_list, seq2_list):
        V1_seq = []
        V2_seq = []
        for d1, d2 in zip(seq1, seq2):
            V1 = np.array([d1.get(k, 0.0) for k in keys], dtype=np.float64)
            V2 = np.array([d2.get(k, 0.0) for k in keys], dtype=np.float64)
            V1_seq.append((ctypes.c_double * len(V1))(*V1))
            V2_seq.append((ctypes.c_double * len(V2))(*V2))
        V1_list.append(V1_seq)
        V2_list.append(V2_seq)

    V1_list_c = (ctypes.POINTER(ctypes.c_double) * num_seqs1)()
    V2_list_c = (ctypes.POINTER(ctypes.c_double) * num_seqs2)()

    for i in range(num_seqs1):
        V1_list_c[i] = (ctypes.POINTER(ctypes.c_double) * len(V1_list[i]))()
        for j in range(len(V1_list[i])):
            V1_list_c[i][j] = V1_list[i][j]

    for i in range(num_seqs2):
        V2_list_c[i] = (ctypes.POINTER(ctypes.c_double) * len(V2_list[i]))()
        for j in range(len(V2_list[i])):
            V2_list_c[i][j] = V2_list[i][j]

    # Call the C function
    Cfunc.fill_matrix(matrix_ptr, V1_list_c, V2_list_c, num_seqs1, num_seqs2, len(V1_list_c[0]))

    # Set the diagonal entries to zero
    np.fill_diagonal(matrix, 0.0)

    return matrix






@njit(cache=True, nogil=True, fastmath=True)
def kl_divergence(p1, p2):
    """Compute the Kullback–Leibler divergence of discrete probability distributions.

    NOTE: we will not perform error checking in this function because
    this function is used very frequently. The user should check that p1
    and p2 are in fact probability distributions.

    Parameters
    ----------
    p1 : 1d array-like
        probability distribution

    p2 : 1d array-like
        probability distribution

    smooth : float
        amount by which to smooth out the probability distribution of `p2`. This
        is intended to deal with categories with zero probability.

    Returns
    -------
    output : numeric
        kl divergence
    """

    kl_div = (p1 * np.log2(p1 / p2)).sum()

    return kl_div


#@njit(cache=True, nogil=True, fastmath=True)
def js_divergence(p1, p2, smooth=0.0001):
    """Compute the Jensen-Shannon of discrete probability distributions.

    Parameters
    ----------
    p1 : 1d array-like
        probability distribution

    p2 : 1d array-like
        probability distribution

    smooth : float
        amount by which to smooth out the probability distribution of `p2`. This
        is intended to deal with categories with zero probability.

    Returns
    -------
    js_div : numeric
        js divergence
    """

    _P = p1 / norm(p1, ord=1)
    _Q = p2 / norm(p2, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


    
    #p1 = (p1 + smooth) / (1 + smooth)
    #p2 = (p2 + smooth) / (1 + smooth)
    #p = 0.5 * (p1 + p2)
    #js_div = 0.5 * (kl_divergence(p1, p) + kl_divergence(p2, p))
    #if np.isnan(js_div):
    #print(p1,'\n',p2,'\n',np.sqrt(js_div),'#\n')
    return js_div
