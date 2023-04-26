import numpy as np
from numba import njit
from scipy.stats import entropy
from numpy.linalg import norm
import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), 'bin/Cfunc.so')
Cfunc = ctypes.CDLL(lib_path)

#  Define the input type and return type of the C functions
Cfunc.jsd.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
Cfunc.jsd.restype = ctypes.c_double
Cfunc.avg_jsd.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                           ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                           ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
Cfunc.avg_jsd.restype = ctypes.c_double



def theta(seq1_list, seq2_list):
    list_length = len(seq1_list)
    V1_list_c = (ctypes.POINTER(ctypes.c_double) * list_length)()
    V2_list_c = (ctypes.POINTER(ctypes.c_double) * list_length)()
    keylens_c = (ctypes.c_size_t * list_length)()

    V1_list = []
    V2_list = []
    keylens = []
    
    for i in range(list_length):
        keys = set()
        keys = keys.union(seq1_list[i].keys()).union(seq2_list[i].keys())
        V1 = np.array([seq1_list[i].get(k,0.0) for k in keys], dtype=np.float64)
        V2 = np.array([seq2_list[i].get(k,0.0) for k in keys], dtype=np.float64)
        V1_list.append((ctypes.c_double * len(V1))(*V1))
        V2_list.append((ctypes.c_double * len(V2))(*V2))

        V1_list_c[i] = V1_list[i]
        V2_list_c[i] = V2_list[i]

        keylens_c[i] = len(keys)
        
    # Call the C function
    result = Cfunc.avg_jsd(V1_list_c, V2_list_c, list_length, keylens_c)
    return result


Cfunc.avg_jsd_.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t
]
Cfunc.avg_jsd_.restype = ctypes.c_double


def theta_(dict1_list, dict2_list):
    list_length = len(dict1_list)

    V1_list = [list(d.values()) for d in dict1_list]
    key1_list = [list(d.keys()) for d in dict1_list]

    V2_list = [list(d.values()) for d in dict2_list]
    key2_list = [list(d.keys()) for d in dict2_list]

    V1_list_c = (ctypes.POINTER(ctypes.c_double) * list_length)()
    key1_list_c = (ctypes.POINTER(ctypes.POINTER(ctypes.c_char)) * list_length)()
    key1_length = (ctypes.c_size_t * list_length)()

    V2_list_c = (ctypes.POINTER(ctypes.c_double) * list_length)()
    key2_list_c = (ctypes.POINTER(ctypes.POINTER(ctypes.c_char)) * list_length)()
    key2_length = (ctypes.c_size_t * list_length)()

    for i in range(list_length):
        V1_list_c[i] = (ctypes.c_double * len(V1_list[i]))(*V1_list[i])
        V2_list_c[i] = (ctypes.c_double * len(V2_list[i]))(*V2_list[i])

        key1_list_c[i] = (ctypes.POINTER(ctypes.c_char) * len(key1_list[i]))()
        key2_list_c[i] = (ctypes.POINTER(ctypes.c_char) * len(key2_list[i]))()

        key1_length[i] = len(key1_list[i])
        key2_length[i] = len(key2_list[i])

        for j, key in enumerate(key1_list[i]):
            encoded_key = key.encode('utf-8')
            key1_list_c[i][j] = ctypes.POINTER(ctypes.c_char)(ctypes.create_string_buffer(encoded_key))

        for j, key in enumerate(key2_list[i]):
            encoded_key = key.encode('utf-8')
            key2_list_c[i][j] =  ctypes.POINTER(ctypes.c_char)(ctypes.create_string_buffer(encoded_key))

    result = Cfunc.avg_jsd_(V1_list_c, key1_list_c, key1_length, V2_list_c, key2_list_c, key2_length, list_length)

    return result



Cfunc.fill_matrix.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
Cfunc.fill_matrix.restype = None


Cfunc.fill_matrix_symmetric.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
    ctypes.c_int,
    ctypes.c_int
]
Cfunc.fill_matrix_symmetric.restype = None




# the second argument None implies we are
# required to calcluate a symmetric n xn distanec matrix
# if both are not None, we assume we are calculating a
# rectangular matrix in general witj unequal dimensions

def theta_matrix(list_dict1_list, list_dict2_list=None):
    list_length = len(list_dict1_list[0])
    num_seq1 = len(list_dict1_list)
    num_seq2 = num_seq1
    if list_dict2_list is not None:
        num_seq2 = len(list_dict2_list)

    V1_list_c__= (ctypes.POINTER(ctypes.POINTER(ctypes.c_double)) * num_seq1)()
    V2_list_c__= (ctypes.POINTER(ctypes.POINTER(ctypes.c_double)) * num_seq2)()
    key1_list_c__ = (ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char))) * num_seq1)()
    key2_list_c__ = (ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char))) * num_seq2)()

    key1_length__ = (ctypes.POINTER(ctypes.c_size_t) * num_seq1)()
    key2_length__ = (ctypes.POINTER(ctypes.c_size_t) * num_seq2)()


    for r, dict1_list in enumerate(list_dict1_list):
        V1_list = [list(d.values()) for d in dict1_list]
        key1_list = [list(d.keys()) for d in dict1_list]

        V1_list_c = (ctypes.POINTER(ctypes.c_double) * list_length)()
        key1_list_c = (ctypes.POINTER(ctypes.POINTER(ctypes.c_char)) * list_length)()
        key1_length = (ctypes.c_size_t * list_length)()


        for i in range(list_length):
            V1_list_c[i] = (ctypes.c_double * len(V1_list[i]))(*V1_list[i])

            key1_list_c[i] = (ctypes.POINTER(ctypes.c_char) * len(key1_list[i]))()

            key1_length[i] = len(key1_list[i])

            for j, key in enumerate(key1_list[i]):
                encoded_key = key.encode('utf-8')
                key1_list_c[i][j] = ctypes.POINTER(ctypes.c_char)(ctypes.create_string_buffer(encoded_key))

        V1_list_c__[r]=V1_list_c
        key1_list_c__[r]=key1_list_c

        key1_length__[r] = (ctypes.c_size_t * len(key1_length))(*key1_length)

    if list_dict2_list is not None:
        for r, dict2_list in enumerate(list_dict2_list):
            V2_list = [list(d.values()) for d in dict2_list]
            key2_list = [list(d.keys()) for d in dict2_list]

            V2_list_c = (ctypes.POINTER(ctypes.c_double) * list_length)()
            key2_list_c = (ctypes.POINTER(ctypes.POINTER(ctypes.c_char)) * list_length)()
            key2_length = (ctypes.c_size_t * list_length)()


            for i in range(list_length):
                V2_list_c[i] = (ctypes.c_double * len(V2_list[i]))(*V2_list[i])

                key2_list_c[i] = (ctypes.POINTER(ctypes.c_char) * len(key2_list[i]))()

                key2_length[i] = len(key2_list[i])

                for j, key in enumerate(key2_list[i]):
                    encoded_key = key.encode('utf-8')
                    key2_list_c[i][j] = ctypes.POINTER(ctypes.c_char)(ctypes.create_string_buffer(encoded_key))

            V2_list_c__[r]=V2_list_c
            key2_list_c__[r]=key2_list_c
            key2_length__[r] = (ctypes.c_size_t * len(key2_length))(*key2_length)
 

 # Create the matrix
    matrix = np.empty((num_seq1, num_seq2), dtype=np.float64)

    if list_dict2_list is None:
        np.fill_diagonal(matrix, 0)
        
    matrix_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if list_dict2_list is None:
        Cfunc.fill_matrix_symmetric(matrix_ptr,V1_list_c__,
                                    key1_list_c__, key1_length__,
                                    list_length,
                                    num_seq1)
    else:
        Cfunc.fill_matrix(matrix_ptr,V1_list_c__,
                          key1_list_c__, key1_length__,
                          V2_list_c__, key2_list_c__,
                          key2_length__, list_length,
                          num_seq1,num_seq2)
        

    return matrix


import concurrent.futures
def process_dict1_list(dict1_list):
    list_length = len(dict1_list)
    V1_list = [list(d.values()) for d in dict1_list]
    key1_list = [list(d.keys()) for d in dict1_list]
    return V1_list, key1_list

def process_dict2_list(dict2_list):
    list_length = len(dict2_list)
    V2_list = [list(d.values()) for d in dict2_list]
    key2_list = [list(d.keys()) for d in dict2_list]
    return V2_list, key2_list

def convert_lists_to_ctypes(V_list, key_list):
    list_length = len(V_list)
    V_list_c = (ctypes.POINTER(ctypes.c_double) * list_length)()
    key_list_c = (ctypes.POINTER(ctypes.POINTER(ctypes.c_char)) * list_length)()
    key_length = (ctypes.c_size_t * list_length)()

    for i in range(list_length):
        V_list_c[i] = (ctypes.c_double * len(V_list[i]))(*V_list[i])
        key_list_c[i] = (ctypes.POINTER(ctypes.c_char) * len(key_list[i]))()
        key_length[i] = len(key_list[i])

        for j, key in enumerate(key_list[i]):
            encoded_key = key.encode('utf-8')
            key_list_c[i][j] = ctypes.POINTER(ctypes.c_char)(ctypes.create_string_buffer(encoded_key))

    return V_list_c, key_list_c, key_length

def theta_matrix_par(list_dict1_list, list_dict2_list):
    num_seq1 = len(list_dict1_list)
    num_seq2 = len(list_dict2_list)
    list_length = len(list_dict1_list[0])
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results1 = list(executor.map(process_dict1_list, list_dict1_list))
        results2 = list(executor.map(process_dict2_list, list_dict2_list))

    V1_list_c__, key1_list_c__, key1_length__ = zip(*[convert_lists_to_ctypes(V_list, key_list) for V_list, key_list in results1])
    V2_list_c__, key2_list_c__, key2_length__ = zip(*[convert_lists_to_ctypes(V_list, key_list) for V_list, key_list in results2])

    V1_list_c__ = (ctypes.POINTER(ctypes.POINTER(ctypes.c_double)) * num_seq1)(*V1_list_c__)
    V2_list_c__ = (ctypes.POINTER(ctypes.POINTER(ctypes.c_double)) * num_seq2)(*V2_list_c__)
    key1_list_c__ = (ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char))) * num_seq1)(*key1_list_c__)
    key2_list_c__ = (ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_char))) * num_seq2)(*key2_list_c__)
    key1_length__ = (ctypes.POINTER(ctypes.c_size_t) * num_seq1)(*key1_length__)
    key2_length__ = (ctypes.POINTER(ctypes.c_size_t) * num_seq2)(*key2_length__)

    # Create the matrix
    matrix = np.empty((num_seq1, num_seq2), dtype=np.float64)
    np.fill_diagonal(matrix, 0)
    matrix_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
    Cfunc.fill_matrix(matrix_ptr, V1_list_c__,
                      key1_list_c__, key1_length__,
                      V2_list_c__, key2_list_c__,
                      key2_length__, list_length,
                      num_seq1, num_seq2)

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
