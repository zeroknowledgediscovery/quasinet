from decimal import Decimal

# from numba import autojit
import numpy as np

# from externals.six.moves import range


def powerset(s):
    """Get the power set of a list or set.
    """
    
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


def remove_zeros(r, axis):
    """Remove rows along a certain axis where the value is all zero.
    """

    if axis == 0:
        return r[~np.all(r == 0, axis=1)]
    elif axis == 1:
        return r[:, ~np.all(r == 0, axis=0)]
    else:
        raise ValueError('not a correct axis that we can use.')

def scientific_notation(num):
    """Convert a number into scientific notation

    Parameters
    ----------
    num : float
        Any number
    
    Returns
    -------
    output : str
        String representation of the number
    """

    return "{:.2E}".format(Decimal(num))

def bayes_boot_probs(n):
    """Bayesian bootstrap sampling for case weights
    
    Parameters
    ----------
    n : int
        Number of Bayesian bootstrap samples
    
    Returns
    -------
    p : 1d array-like
        Array of sampling probabilities
    """
    p = np.random.exponential(scale=1.0, size=n)
    return p/p.sum()


# @autojit(nopython=True, cache=True, nogil=True)
def auc_score(y_true, y_prob):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    y_true, n   = y_true[np.argsort(y_prob)], len(y_true)
    nfalse, auc = 0, 0.0
    for i in range(n):
        nfalse += 1 - y_true[i]
        auc    += y_true[i] * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def logger(name, message):
    """Prints messages with style "[NAME] message"
    
    Parameters
    ----------
    name : str
        Short title of message, for example, train or test

    message : str
        Main description to be displayed in terminal
    
    Returns
    -------
    None
    """
    print('[{name}] {message}'.format(name=name.upper(), message=message))


def estimate_margin(y_probs, y_true):
    """Estimates margin function of forest ensemble

    Note : This function is similar to margin in R's randomForest package
    
    Parameters
    ----------
    y_probs : 2d array-like
        Predicted probabilities where each row represents predicted
        class distribution for sample and each column corresponds to 
        estimated class probability

    y_true : 1d array-like
        Array of true class labels
    
    Returns
    -------
    margin : float
        Estimated margin of forest ensemble
    """
    # Calculate probability of correct class
    n, p        = y_probs.shape
    true_probs  = y_probs[np.arange(n, dtype=int), y_true]

    # Calculate maximum probability for incorrect class
    other_probs = np.zeros(n)
    for i in range(n):
        mask            = np.zeros(p, dtype=bool)
        mask[y_true[i]] = True
        other_idx       = np.ma.array(y_probs[i,:], mask=mask).argmax()
        other_probs[i]  = y_probs[i, other_idx]
    
    # Margin is P(y == j) - max(P(y != j))
    return true_probs - other_probs



def assert_array_rank(X, rank):
    """Check if the input is an numpy array and has a certain rank.

    Parameters
    ----------
    X : array-like
        Array to check

    rank : int
        Rank of the tensor to check
    
    Returns
    -------
    None
    """

    if not isinstance(X, np.ndarray):
        raise ValueError('You must pass in a numpy array!')

    if len(X.shape) != rank:
        raise ValueError('You must pass in a {}-rank array!'.format(rank))


def sample_from_dict(distrib):
    """Choose an item from the distribution

    Parameters
    ----------
    distrib : dict
        dictionary mapping keys to its probability values

    Returns
    -------
    item : key of dict
        a chosen key from the dictionary
    """

    keys = []
    probs = []
    for k, prob in distrib.items():
        keys.append(k)
        probs.append(prob)

    item = np.random.choice(keys, p=probs)

    return item
