import numpy as np
from numba import njit

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

    # p1 += smooth
    # p1 /= np.sum(p1)

    # p2 += smooth
    # p2 /= np.sum(p2)

    kl_div = (p1 * np.log2(p1 / p2)).sum()

    return kl_div


@njit(cache=True, nogil=True, fastmath=True)
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
    
    # TODO: this checking may cost us a lot of unneccesary computation time
    if np.all(p1 == p2):
        return 0.0
    else:
        p1 = (p1 + smooth) / (1 + smooth)
        p2 = (p2 + smooth) / (1 + smooth)
        p = 0.5 * (p1 + p2)
        js_div = 0.5 * (kl_divergence(p1, p) + kl_divergence(p2, p))

        return js_div