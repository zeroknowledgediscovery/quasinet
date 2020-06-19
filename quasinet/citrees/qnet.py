
import pandas as pd
import numpy as np
from joblib import dump, load, delayed, Parallel

from citrees import CITreeClassifier
from metrics import js_divergence
from tree import Node, get_nodes

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
        output : dictionary
            dictionary mapping column names to another dictionary. That dictionary
            maps possible column values to probability values.
        """

        self._check_is_fitted()

        col_to_prob_distrib = {}
        for col in self.estimators_.keys():
            column_to_item = {i: item for i, item in enumerate(seq)}
            col_to_prob_distrib[col] = self.predict_distribution(column_to_item, col)

        return col_to_prob_distrib


    def predict_proba(self, X):

        self._check_is_fitted()

        raise NotImplementedError



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

    total_divergence = 0
    for col, seq1_distrib in seq1_distribs.items():

        seq2_distrib = seq2_distribs[col]

        distrib = _combine_two_distribs(seq1_distrib, seq2_distrib)

        total_divergence += np.sqrt(js_divergence(distrib[0], distrib[1], smooth=True))

    total_divergence /= len(seq1_distribs)
    
    return total_divergence


def qdistance_with_prob_distributions(distrib1, distrib2):
    raise NotImplementedError

def qdistance_matrix(seqs1, seqs2, qnet1, qnet2):
    raise NotImplementedError

def load_qnet(f):
    qnet = load(f)
    assert isinstance(qnet, Qnet)

    return qnet

def save_qnet(qnet, f):
    assert isinstance(qnet, Qnet)
    if not f.endswith('.joblib'):
        raise ValueError('The outfile must end with `.joblib`')

    dump(qnet, f) 