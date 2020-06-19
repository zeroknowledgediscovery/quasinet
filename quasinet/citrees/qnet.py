
import pandas as pd
import numpy as np
from joblib import dump, load, delayed, Parallel

from citrees import CITreeClassifier
from metrics import js_divergence
from tree import Node, get_nodes
import numba
from numba import njit, prange
from numba.core import types

class Qnet(object):
    """
    """

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def __repr__(self):
        return "qnet.Qnet"

    def __str__(self):
        return self.__repr__()

    def _parallel_fit_tree(self, tree, X, col):
        tree.fit(np.delete(X, col, 1), X[:, col])

        return tree

    def fit(self, X):

        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise ValueError('X must be a 2d numpy array!')

        if not np.issubdtype(X.dtype, np.str_):
            raise ValueError('X must contain only strings!')
            

        # Instantiate base tree models
        self.estimators_ = {}

        # TODO: allow for more arguments to be passed to CITrees
        # TODO: we may not have any trees created. When that's the
        # case, we want to predict an equal probability distribution

        trees = []
        for col in np.arange(0, X.shape[1]):
            tree = CITreeClassifier(selector='chi2', random_state=42)
            trees.append(tree)

        trees = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(self._parallel_fit_tree)(
                trees[col], X, col)
            for col in range(0, X.shape[1])
            )

        for col, tree in enumerate(trees):
            self.estimators_[col] = tree

            # clf.fit(np.delete(X, col, 1), X[:, col])
            # self.estimators_[col] = clf

        return self


    def _check_is_fitted(self):
        if not hasattr(self, 'estimators_'):
            raise ValueError('You need to call `fit` first! ')


    def predict_distribution(self, column_to_item, column):
        """Predict the probability distribution for a given column.

        Parameters
        ----------
        column_to_item : dict
            dictionary mapping the column to the values the columns take

        column : str
            column name

        Returns
        -------
        output : dictionary
            dictionary mapping possible column values to probability values
        """

        self._check_is_fitted()

        if len(column_to_item) == 0:
            raise NotImplementedError

        root = self.estimators_[column].root
        nodes = get_nodes(root)
        distributions = {}
        for node in nodes:
            if node.col in column_to_item:
                if column_to_item[node.col] in node.threshold:
                    next_node = node.left
                else:
                    next_node = node.right

                distributions[node.col] = next_node.label_frequency

        if len(distributions) == 0:
            distributions[root.col] = root.label_frequency

        # TODO: eventually, it may be the case that we have to remove
        # all usages of pandas because numba does not allow for pandas
        distributions = pd.DataFrame(list(distributions.values()))
        distributions.fillna(0, inplace=True)
        distributions.reset_index(inplace=True, drop=True)
  
        total_frequency = np.sum(distributions.values)        
        distributions = distributions.mul(distributions.sum(axis=1), axis=0)
        distributions /= total_frequency

        distributions = distributions.mean(axis=0)
        distributions /= np.sum(distributions.values)

        return distributions.to_dict()



    def predict_distributions(self, seq):
        """Predict the probability distributions for all the columns.

        Parameters
        ----------
        seq : list
            list of values

        Returns
        -------
        prob_distributions : list
            list of probability distributions, one for each index 
        """

        self._check_is_fitted()

        prob_distributions = [None] * len(self.estimators_)
        for col in self.estimators_.keys():
            column_to_item = {i: item for i, item in enumerate(seq)}
            distrib = self.predict_distribution(column_to_item, col)
            prob_distributions[col] = distrib

        return prob_distributions

    def predict_distributions_numba(self, seq):
        prob_distributions = self.predict_distributions(seq)

        numba_prob_distributions = []
        for prob_distribution in prob_distributions:
            d = numba.typed.Dict.empty(
                key_type=types.unicode_type,
                value_type=types.float32)

            for k, v in prob_distribution.items():
                d[k] = v
            numba_prob_distributions.append(d)

        return numba_prob_distributions

    def predict_proba(self, X):

        self._check_is_fitted()

        raise NotImplementedError


@njit(cache=True, nogil=True, fastmath=True)
def _combine_two_distribs(seq1_distrib, seq2_distrib):
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

    # make sure both distribution contains the same responses
    for seq1_response in seq1_distrib.keys():
        if seq1_response not in seq2_distrib:
            seq2_distrib[seq1_response] = 0.0

    for seq2_response in seq2_distrib.keys():
        if seq2_response not in seq1_distrib:
            seq1_distrib[seq2_response] = 0.0

    num_responses = len(seq2_distrib)

    distrib = np.empty((2, num_responses))
    for i, x in enumerate(seq1_distrib.keys()):
        distrib[0, i] = seq1_distrib[x]
        distrib[1, i] = seq2_distrib[x]

    return distrib


@njit(cache=True, nogil=True, fastmath=True)
def _qdistance_with_prob_distribs(distrib1, distrib2):
    """using njit worsens speed performance
    """

    total_divergence = 0

    for i, seq1_distrib in enumerate(distrib1):

        seq2_distrib = distrib2[i]

        distrib = _combine_two_distribs(seq1_distrib, seq2_distrib)

        total_divergence += np.sqrt(js_divergence(distrib[0], distrib[1]))

    avg_divergence = total_divergence / len(distrib1)

    return avg_divergence


def qdistance(seq1, seq2, qnet1, qnet2):
    """Compute the Jensen-Shannon of discrete probability distributions.

    Parameters
    ----------
    seq1 : 1d array-like
        Array of values

    seq2 : 1d array-like
        Array of values

    qnet1 : Qnet
        the Qnet that `seq1` belongs to 

    qnet2 : Qnet
        the Qnet that `seq2` belongs to 

    Returns
    -------
    output : numeric
        qdistance
    """

    if not isinstance(seq1, np.ndarray) or not isinstance(seq2, np.ndarray):
        raise ValueError('You must pass in arrays as sequences.')

    if len(seq1.shape) != 1 or len(seq2.shape) != 1:
        raise ValueError('The two sequences must be 1d arrays.')

    if seq1.shape[0] != seq2.shape[0]:
        raise ValueError('The two sequences must be of equal lengths.')

    seq1_distribs = qnet1.predict_distributions(seq1)
    seq2_distribs = qnet2.predict_distributions(seq2)
    
    divergence = _qdistance_with_prob_distribs(seq1_distribs, seq2_distribs)
    return divergence


@njit(parallel=True, fastmath=True)
# @njit
def _qdistance_matrix_with_distribs(seqs1_distribs, seqs2_distribs):
    num_seqs1 = len(seqs1_distribs)
    num_seqs2 = len(seqs2_distribs)

    distance_matrix = np.empty((num_seqs1, num_seqs2))
    # for i in np.arange(num_seqs1):
    for i in prange(num_seqs1):
        for j in np.arange(num_seqs2):
            distance_matrix[i, j] = _qdistance_with_prob_distribs(seqs1_distribs[i], 
                                                                  seqs2_distribs[j])

    return distance_matrix

def qdistance_matrix(seqs1, seqs2, qnet1, qnet2):

    seqs1_distribs = numba.typed.List()
    for seq in seqs1:
        seqs1_distribs.append(qnet1.predict_distributions_numba(seq))

    seqs2_distribs = numba.typed.List()
    for seq in seqs2:
        seqs2_distribs.append(qnet2.predict_distributions_numba(seq))

    # seqs1_distribs = [qnet1.predict_distributions_numba(seq) for seq in seqs1]
    # seqs2_distribs = [qnet2.predict_distributions_numba(seq) for seq in seqs2]

    distance_matrix = _qdistance_matrix_with_distribs(seqs1_distribs, seqs2_distribs)

    return distance_matrix
    

def load_qnet(f):
    qnet = load(f)
    assert isinstance(qnet, Qnet)

    return qnet

def save_qnet(qnet, f):
    assert isinstance(qnet, Qnet)
    if not f.endswith('.joblib'):
        raise ValueError('The outfile must end with `.joblib`')

    dump(qnet, f) 