import numpy as np

def kl_divergence(p1, p2, smooth=True):
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

    smooth : bool
        whether to smooth out the probability distribution of `p2`. This
        is intended to deal with categories with zero probability.

    Returns
    -------
    output : numeric
        kl divergence
    """

    if smooth:
        p2 = (p2 + 0.0001)
        p2 /= np.sum(p2)

    return (p1 * np.log2(p1 / p2)).sum()


def js_divergence(p1, p2, smooth=True):
    """Compute the Jensen-Shannon of discrete probability distributions.

    Parameters
    ----------
    p1 : 1d array-like
        probability distribution

    p2 : 1d array-like
        probability distribution

    smooth : bool
        whether to smooth out the probability distributions. 

    Returns
    -------
    output : numeric
        js divergence
    """
    
    p = 0.5 * (p1 + p2)
    return 0.5 * (kl_divergence(p1, p, smooth) + kl_divergence(p2, p, smooth))