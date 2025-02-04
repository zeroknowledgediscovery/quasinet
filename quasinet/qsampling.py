import random
import warnings

import numpy as np

from .utils import assert_array_rank, sample_from_dict, assert_string_type, generate_seed
from ._config import get_config

# def _qsample_with_prob_distribs(seq, distrib):

def _qsample_once(seq, qnet, baseline_prob, force_change, alpha=None,RNG=None):
    """Perform one instance of q-sampling.

    NOTE: `seq` is modified in-place

    Parameters
    ----------
    seq : 1d array-like
        Array of values

    qnet : Qnet
        The Qnet that `seq` belongs to 

    baseline_prob : 1d array-like
        Baseline probability for sampling which index

    force_change : bool
        Whether to force the sequence to change when sampling. 
        There is a special cases where the predicted distribution is a single 
        item with 100% probability and the item is exactly the item in the
        corresponding index of `seq`. Then in that case, we are unable to
        force the sequence to change and we simply return the original sequence.


    Returns
    -------
    None
    """

    seq_len = len(seq)

    if seq_len != len(qnet.estimators_):
        message = ('The input sequence length ({}) must match the' 
                  'number of features trained on the qnet ({})')
        warnings.warn(message.format(seq_len, len(qnet.estimators_)))

    # get the index distribution from a distribution
    if baseline_prob is None:
        if RNG:
            index = RNG.integers(0, seq_len, endpoint=False)
        else:
            index = np.random.randint(0,seq_len)
    else:
        if RNG:
            index = RNG.choice(
                np.arange(0, seq_len),
                p=baseline_prob)
        else:
            index = np.random.choice(
                np.arange(0, seq_len),
                p=baseline_prob)            

    # get distribution corresponding to the index
    index_to_non_leaf_nodes = qnet._map_col_to_non_leaf_nodes()
    col_to_item = {i: seq[i] for i in index_to_non_leaf_nodes[index]}
    distrib = qnet.predict_distribution(col_to_item, index)

    if alpha:
        d_ = {k:(v**alpha) for k,v in distrib.items()}
        S_=np.array([v for k,v in d_.items()]).sum()
        if S_ > 0.0:
            distrib = {k:v/S_ for k,v in d_.items()}
            

    if force_change:
        if seq[index] in distrib:
            del distrib[seq[index]]

        if len(distrib) == 0:
            return

        if np.sum(list(distrib.values())) == 0:
            return

    item = sample_from_dict(distrib)

    seq[index] = item


def qsample(seq, qnet, steps, baseline_prob=None,
            force_change=False, alpha=None, random_seed=None):
    """Perform q-sampling for multiple steps or specified indices.

    Qsampling works as follows: Say you have a sequence and a qnet. Then 
    we randomly pick one of the items in the sequence (or a specified index)
    and change the value of that item based on the prediction of the qnet.

    Parameters
    ----------
    seq : 1d array-like
        Array of values

    qnet : Qnet
        The Qnet that `seq` belongs to 

    steps : int or 1d array-like
        If an integer, the number of steps to run q-sampling.
        If an array, specifies the indices to q-sample in order.

    baseline_prob : 1d array-like, optional
        Baseline probability for sampling which index. Ignored if `steps` is an array.

    force_change : bool, optional
        Whether to force the sequence to change when sampling. 

    alpha : float, optional
        Scalar multiple of qnet object, can be any real number.

    random_seed : int, optional
        Seed for reproducible randomness.

    Returns
    -------
    seq : 1d array-like
        q-sampled sequence
    """

    assert_array_rank(seq, 1)
    assert_string_type(seq, 'seq')

    if baseline_prob is not None:
        assert_array_rank(baseline_prob, 1)

    if random_seed:
        RNG = np.random.default_rng(random_seed)
    else:
        RNG = None

    seq = seq.copy()

    if isinstance(steps, int):
        # Perform q-sampling for a fixed number of random steps
        for _ in range(steps):
            _qsample_once(seq, qnet, baseline_prob,
                          force_change=force_change, alpha=alpha, RNG=RNG)
    elif isinstance(steps, (list, np.ndarray)):
        # Perform q-sampling for specific indices in `steps`
        steps = np.asarray(steps)
        for idx in steps:
            # Create a temporary baseline_prob focusing only on the given index
            temp_baseline_prob = np.zeros(len(seq))
            temp_baseline_prob[idx] = 1.0
            _qsample_once(seq, qnet, temp_baseline_prob,
                          force_change=force_change, alpha=alpha, RNG=RNG)
    else:
        raise ValueError("`steps` must be an integer or a 1D array-like of indices.")

    return seq


    
#def qsample(seq, qnet, steps, baseline_prob=None,
#            force_change=False, alpha=None, random_seed=None):
#    """Perform q-sampling for multiple steps.#
#
#    Qsampling works as follows: Say you have a sequence and a qnet. Then 
#    we randomly pick one of the items in the sequence and then change the
#    value of that item based on the prediction of the qnet.
#
#    Parameters
#    ----------
#    seq : 1d array-like
#        Array of values
#
#    qnet : Qnet
#        The Qnet that `seq` belongs to 
#
#    steps : int
#        Number of steps to run q-sampling
#
#    baseline_prob : 1d array-like
#        Baseline probability for sampling which index
#
#    force_change : bool
#        Whether to force the sequence to change when sampling. 
#
#    alpha : float
#        scalr multiple of qnet object, can be any real number
#
#    Returns
#    -------
#    seq : 1d array-like
#        q-sampled sequence
#    """
#
#    assert_array_rank(seq, 1)
#    assert_string_type(seq, 'seq')
#
#    if baseline_prob is not None:
#        assert_array_rank(baseline_prob, 1)
#
#    
#    if random_seed:
#        seed = generate_seed()
#        RNG = np.random.default_rng(seed)
#    else:
#        RNG= None
#        
#    seq = seq.copy()
#    for _ in range(steps):
#        _qsample_once(
#            seq, 
#            qnet, 
#            baseline_prob,
#            force_change=force_change,alpha=alpha,RNG=RNG)
#
#    return seq

def targeted_qsample(seq1, seq2, qnet, steps, force_change=False):
    """Perform targeted q-sampling for multiple steps.

    `seq1` is q-sampled towards `seq2`.

    This is similar to `qsample`, except that we perform changes to `seq1`
    to try to approach `seq2`.

    Parameters
    ----------
    seq1 : 1d array-like
        Array of values

    seq2 : 1d array-like
        Array of values. 

    qnet : Qnet
        The Qnet that `seq1` belongs to 

    steps : int
        Number of steps to run q-sampling

    force_change : bool
        Whether to force the sequence to change when sampling. 

    Returns
    -------
    seq : 1d array-like
        q-sampled sequence
    """

    assert_array_rank(seq1, 1)
    assert_array_rank(seq2, 1)
    assert_string_type(seq1, 'seq1')
    assert_string_type(seq2, 'seq2')

    if seq1.shape[0] != seq2.shape[0]:
        raise ValueError('The lengths of the two sequences must be equal!')

    nan_value = get_config()['nan_value']
    seq = seq1.copy()
    seq_len = seq.shape[0]

    for _ in range(steps):

        # only allow different indexes to be sampled
        probs = np.zeros(seq_len, dtype=np.float32)
        probs[seq != seq2] = 1.0
        probs[seq2 == nan_value] = 0.0

        probs /= np.sum(probs)

        _qsample_once(
            seq, 
            qnet, 
            baseline_prob=probs,
            force_change=force_change)

    return seq
