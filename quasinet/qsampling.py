import random

import numpy as np

from .utils import assert_array_rank, sample_from_dict

# def _qsample_with_prob_distribs(seq, distrib):

def _qsample_once(seq, qnet, baseline_prob):
    """Perform one instance of q-sampling.

    NOTE: `seq` is modified in-place
    """

    seq_distribs = qnet.predict_distributions(seq)

    seq_distribs_len = len(seq_distribs)

    # choose a distribution
    if baseline_prob is None:
        index = np.random.randint(0, seq_distribs_len)
    else:
        index = np.random.choice(
            np.arange(0, seq_distribs_len),
            p=baseline_prob)

    distrib = seq_distribs[index]

    item = sample_from_dict(distrib)

    seq[index] = item

def qsample(seq, qnet, steps, baseline_prob=None):
    """Perform q-sampling for multiple steps.

    Parameters
    ----------
    seq : 1d array-like
        Array of values

    qnet : Qnet
        the Qnet that `seq` belongs to 

    steps : int
        number of steps to run q-sampling

    baseline_prob : 1d array-like
        Baseline probability for sampling which index

    Returns
    -------
    seq : 1d array-like
        q-sampled sequence
    """

    assert_array_rank(seq, 1)
    if baseline_prob is not None:
        assert_array_rank(baseline_prob, 1)

    seq = seq.copy()
    for _ in range(steps):
        _qsample_once(seq, qnet, baseline_prob)

    return seq

def targeted_qsample(seq1, seq2, qnet, steps):
    """Perform q-sampling for multiple steps.

    `seq1` is q-sampled towards `seq2`.

    Parameters
    ----------
    seq1 : 1d array-like
        Array of values

    seq2 : 1d array-like
        Array of values. 

    qnet : Qnet
        the Qnet that `seq1` belongs to 

    steps : int
        number of steps to run q-sampling

    Returns
    -------
    seq : 1d array-like
        q-sampled sequence
    """

    assert_array_rank(seq1, 1)
    assert_array_rank(seq2, 1)

    if seq1.shape[0] != seq2.shape[0]:
        raise ValueError('The lengths of the two sequences must be equal!')

    seq = seq1.copy()
    seq_len = seq.shape[0]

    for _ in range(steps):

        # only allow different indexes to be sampled
        probs = np.zeros(seq_len, dtype=np.float32)
        for i in range(seq_len):
            if seq[i] != seq2[i]:
                probs[i] = 1.0

        probs /= np.sum(probs)

        _qsample_once(seq, qnet, baseline_prob=probs)

    return seq