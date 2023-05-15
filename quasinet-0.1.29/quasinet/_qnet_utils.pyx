cimport numpy as np
import numpy as np

ctypedef np.float32_t float32
ctypedef np.int32_t int32
ctypedef np.int8_t int8

def _combine_two_distribs(dict seq1_distrib, dict seq2_distrib):
    """Combine two distributions together.

    The two distributions may contain different random variables.
    If one random variable doesn't exist in one of the distributions,
    then we set the probability to 0.

    Parameters
    ----------
    seq1_distrib : dict
        Dictionary mapping random variables to probabilities.

    seq2_distrib : dict
        Dictionary mapping random variables to probabilities.

    Returns
    -------
    output : 2d array-like
        array of shape (2, total unique random variables)
    """

    cdef float32 float32_zero = 0.0
    cdef np.ndarray[float32, ndim=2] distrib
    cdef str seq1_response, seq2_response, x
    cdef int32 i

    # make sure both distribution contains the same responses
    for seq1_response in seq1_distrib.keys():
        if seq1_response not in seq2_distrib:
            seq2_distrib[seq1_response] = float32_zero

    for seq2_response in seq2_distrib.keys():
        if seq2_response not in seq1_distrib:
            seq1_distrib[seq2_response] = float32_zero

    distrib = np.empty((2, len(seq2_distrib)), dtype=np.float32)
    for i, x in enumerate(seq1_distrib.keys()):
        distrib[0, i] = seq1_distrib[x]
        distrib[1, i] = seq2_distrib[x]

    return distrib