
import numpy as np
from joblib import dump, load, delayed, Parallel
import numba
# from numba import njit, prange
from numba.core import types

from .citrees import CITreeClassifier
from .metrics import js_divergence
from .tree import Node, get_nodes
from .utils import assert_array_rank
from ._export import GraphvizExporter
from ._config import get_config

class Qnet(object):
    """Qnet architecture.

    Parameters
    ----------

    feature_names : list
        List of names describing the features

    min_samples_split : int
        Minimum samples required for a split

    alpha : float
        Threshold value for selecting feature with permutation tests. Smaller
        values correspond to shallower trees

    max_depth : int
        Maximum depth to grow tree

    max_feats : str or int
        Maximum feats to select at each split. String arguments include 'sqrt',
        'log', and 'all'

    early_stopping : bool
        Whether to implement early stopping during feature selection. If True,
        then as soon as the first permutation test returns a p-value less than
        alpha, this feature will be chosen as the splitting variable

    verbose : bool or int
        Controls verbosity of training and testing

    random_state : int
        Sets seed for random number generator

    n_jobs : int
        Number of CPUs to use when training
    """

    def __init__(self, 
                 feature_names,
                 min_samples_split=2,
                 alpha=.05,
                 max_depth=-1,
                 max_feats=-1,
                 early_stopping=False,
                 verbose=0,
                 random_state=None,
                 n_jobs=1):

        self.feature_names = feature_names
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.max_depth = max_depth
        self.max_feats = max_feats
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

    def __repr__(self):
        return "qnet.Qnet"

    def __str__(self):
        return self.__repr__()

    def _parallel_fit_tree(self, tree, X, col):
        new_features = np.copy(X)
        new_features[:, col] = get_config()['nan_value']
        tree.fit(new_features, X[:, col])

        return tree

    def _check_input_size(self, size):
        feature_size = len(self.feature_names)
        if size != feature_size:
            string = 'The number of input features ({}) ' + \
                     'must match size of `feature_names` ({})!'
            string = string.format(size, feature_size)
            raise ValueError(string)

    def _check_is_fitted(self):
        if not hasattr(self, 'estimators_'):
            raise ValueError('You need to call `fit` first! ')

    def fit(self, X):

        assert_array_rank(X, 2)
        
        self._check_input_size(X.shape[1])

        if not np.issubdtype(X.dtype, np.str_):
            raise ValueError('X must contain only strings!')
            

        # Instantiate base tree models
        self.estimators_ = {}

        # TODO: we may not have any trees created. When that's the
        # case, we want to predict an equal probability distribution

        trees = []
        for col in np.arange(0, X.shape[1]):
            tree = CITreeClassifier(min_samples_split=self.min_samples_split,
                                    alpha=self.alpha,
                                    selector='chi2',
                                    max_depth=self.max_depth,
                                    max_feats=self.max_feats,
                                    early_stopping=self.early_stopping,
                                    verbose=self.verbose,
                                    random_state=self.random_state)
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



    def _map_col_to_non_leaf_nodes(self):
        """Get a mapping from column indexes to an array of all 
        non-leaf node indexes from the corresponding tree.

        We will cache the results because this function is called
        multiple times and it doesn't depend on any input argument.

        Parameters
        ----------
        None

        Returns
        -------
        prob_distributions : dictionary
            dictionary mapping indexes to a list of non-leaf nodes.
        """

        if hasattr(self, 'col_to_non_leaf_nodes'):
            pass
        else:
            self._check_is_fitted()
            
            col_to_nodes = {}
            for col, clf_tree in self.estimators_.items():
                nodes = get_nodes(
                    clf_tree.root, 
                    get_leaves=False, get_non_leaves=True)
                
                node_cols = np.array([node.col for node in nodes])

                col_to_nodes[col] = np.unique(node_cols)
            self.col_to_non_leaf_nodes = col_to_nodes

        return self.col_to_non_leaf_nodes


    def clear_attributes(self):
        """Remove the unneeded attributes to save memory.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self._check_is_fitted()

        if hasattr(self, 'col_to_non_leaf_nodes'):
            delattr(self, 'col_to_non_leaf_nodes')

        new_estimator = {}
        for col, tree in self.estimators_.items():
            new_tree = CITreeClassifier(min_samples_split=self.min_samples_split,
                                        alpha=self.alpha,
                                        selector='chi2',
                                        max_depth=self.max_depth,
                                        max_feats=self.max_feats,
                                        early_stopping=self.early_stopping,
                                        verbose=self.verbose,
                                        random_state=self.random_state)

            new_tree.root = tree.root
            new_tree.labels_ = tree.labels_
            new_estimator[col] = new_tree

        delattr(self, 'estimators_')

        self.estimators_ = new_estimator

    def _combine_distributions(self, distributions):
        """Given a list of distributions, combine them together into
        an array.

        Parameters
        ----------
        distributions : list
            list of dictionaries mapping responses to probabilities.

        Returns
        -------
        array : 2d array-like
            the resulting array from combining the distributions.

        all_responses : list
            list of the responses corresponding to the array
        """

        all_responses = []
        for distrib in distributions:
            all_responses.extend(distrib.keys())

        all_responses = list(set(all_responses))

        res_to_index = {res: i for i, res in enumerate(all_responses)}

        array = np.zeros((len(distributions), len(all_responses)))

        for i, distrib in enumerate(distributions):
            for k, v in distrib.items():

                array[i, res_to_index[k]] = v

        return array, all_responses

    def predict_distribution(self, column_to_item, column):
        """Predict the probability distribution for a given column.

        It may be the case that a certain column value does not appear in 
        the resulting output. If that happens, that means the probability
        of that column is 0.

        Parameters
        ----------
        column_to_item : dict
            dictionary mapping the column to the values the columns take

        column : int
            column index

        Returns
        -------
        output : dictionary
            dictionary mapping possible column values to probability values
        """

        self._check_is_fitted()

        root = self.estimators_[column].root

        if len(column_to_item) == 0:
            distributions = dict(root.label_frequency)
            values = float(sum(distributions.values()))
            distributions = {k: v / values for k, v in distributions.items()}
            return distributions

        nodes = get_nodes(root)
        distributions = []
        for node in nodes:
            if node.col in column_to_item:
                if column_to_item[node.col] in node.lthreshold:
                    next_node = node.left
                elif column_to_item[node.col] in node.rthreshold:
                    next_node = node.right
                else:
                    continue

                distributions.append(next_node.label_frequency)

        if len(distributions) == 0:
            distributions = [root.label_frequency]

        distributions, columns = self._combine_distributions(distributions)
        
        total_frequency = np.sum(distributions)        
        distributions = distributions * np.sum(distributions, axis=1)[:, None]
        distributions /= total_frequency

        distributions = np.mean(distributions, axis=0)
        distributions /= np.sum(distributions)

        output = {columns[i]: distributions[i] for i in range(len(columns))}
        return output


    def predict_distributions(self, seq):
        """Predict the probability distributions for all the columns.

        If you do not want to set a particular value for an index of `seq`, 
        then set the value at the index to the global `nan_value`. By default, 
        this value is the empty string.

        The length of the input sequence must match the size of `feature_names`.

        Parameters
        ----------
        seq : list
            list of values

        Returns
        -------
        prob_distributions : list
            list of dictionaries of probability distributions, one for each index 
        """

        self._check_is_fitted()
        self._check_input_size(len(seq))

        index_to_non_leaf_nodes = self._map_col_to_non_leaf_nodes()

        prob_distributions = [None] * (max(self.estimators_.keys()) + 1)
        for col in self.estimators_.keys():
            col_to_item = {i: seq[i] for i in index_to_non_leaf_nodes[col]}
            distrib = self.predict_distribution(col_to_item, col)
            prob_distributions[col] = distrib

        return prob_distributions

    def predict_distributions_numba(self, seq):
        raise NotImplementedError
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


def _qdistance_with_prob_distribs(distrib1, distrib2):
    """
    
    NOTE: using njit may worsen speed performance

    Parameters
    ----------
    distrib1 : list
        List of dictionaries. Each dictionary maps random variables 
        to probabilities.

    distrib2 : list
        List of dictionaries. Each dictionary maps random variables 
        to probabilities.

    Returns
    -------
    avg_divergence : scalar
        qdistance
    """

    total_divergence = 0
    total = 0
    for i, seq1_distrib in enumerate(distrib1):
            
        seq2_distrib = distrib2[i]

        if seq2_distrib is None or seq1_distrib is None:
            continue
        
        distrib = _combine_two_distribs(seq1_distrib, seq2_distrib)

        total_divergence += np.sqrt(js_divergence(distrib[0], distrib[1]))

        total += 1
        
    avg_divergence = total_divergence / total

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

def membership_degree(seq, qnet):
    """Compute the membership degree of a sequence in a qnet.

    Parameters
    ----------
    seq : 1d array-like
        Array of values

    qnet : Qnet
        the Qnet that `seq` belongs to 

    Returns
    -------
    membership_degree : numeric
        membership degree
    """

    assert_array_rank(seq, 1)

    if len(seq) == 0:
        raise ValueError('The sequence cannot be empty.')
    
    seq_distribs = qnet.predict_distributions(seq)

    index_probs = []
    for index, c in enumerate(seq):
        if index not in seq_distribs:
            continue

        distrib = seq_distribs[index]
        if distrib is None:
            continue
        
        if c in distrib:
            index_prob = distrib[c]
            
            if index_prob != 0:
                index_probs.append(index_prob)

    membership_degree = np.sum(np.log(index_probs))

    return membership_degree

def _qdistance_matrix_with_distribs(seqs1_distribs, seqs2_distribs, symmetric):
    """Compute the qdistance matrix using the distributions.

    Parameters
    ----------
    seqs1_distribs : list
        List of lists. Each sublist is a list of dictionaries mapping 
        random variables to probabilities.

    seqs2_distribs : list
        Same as `seqs1_distribs`

    symmetric : bool
        whether the distance matrix is symmetric or not

    Returns
    -------
    distance_matrix : 2d array-like
        distance matrix
    """

    num_seqs1 = len(seqs1_distribs)
    num_seqs2 = len(seqs2_distribs) 

    distance_matrix = np.empty((num_seqs1, num_seqs2))
    for i in np.arange(num_seqs1):
    # for i in prange(num_seqs1):
        for j in np.arange(num_seqs2):

            if symmetric and (i >= j):
                dist = 0.0
            else:
                dist = _qdistance_with_prob_distribs(seqs1_distribs[i], 
                                                     seqs2_distribs[j])

            distance_matrix[i, j] = dist

    if symmetric: 
        distance_matrix = distance_matrix + distance_matrix.T
    return distance_matrix


def qdistance_matrix(seqs1, seqs2, qnet1, qnet2):
    """Compute a distance matrix with the qdistance metric.

    Parameters
    ----------
    seqs1: 2d array-like
        Array of values

    seqs2: 2d array-like
        Array of values

    qnet1 : Qnet
        the Qnet that `seqs1` belongs to 

    qnet2 : Qnet
        the Qnet that `seqs2` belongs to 

    Returns
    -------
    distance_matrix : 2d array-like
        distance matrix
    """

    assert_array_rank(seqs1, 2)
    assert_array_rank(seqs2, 2)

    if seqs1.shape[1] != seqs2.shape[1]:
        raise ValueError('The columns of the two matrices must be equal.')

    symmetric = np.all(seqs1 == seqs2) and (qnet1 == qnet2)

    # WARNING: do not try to access seqs1_distribs with non-numba code, 
    # as it will create a segmentation fault
    # this is likely a bug due to Numba
    # seqs1_distribs = numba.typed.List()
    seqs1_distribs = []
    for seq1 in seqs1:
        seqs1_distribs.append(qnet1.predict_distributions(seq1))

    # seqs2_distribs = numba.typed.List()
    seqs2_distribs = []
    for seq2 in seqs2:
        seqs2_distribs.append(qnet2.predict_distributions(seq2))

    # seqs1_distribs = [qnet1.predict_distributions_numba(seq) for seq in seqs1]
    # seqs2_distribs = [qnet2.predict_distributions_numba(seq) for seq in seqs2]

    distance_matrix = _qdistance_matrix_with_distribs(seqs1_distribs, 
                                                      seqs2_distribs,
                                                      symmetric=symmetric)

    return distance_matrix
    

def load_qnet(f):
    """Load the qnet from a file.

    Parameters
    ----------
    f : str
        File name.
 
    Returns
    -------
    qnet : Qnet
    """

    qnet = load(f)
    assert isinstance(qnet, Qnet)

    return qnet

def save_qnet(qnet, f, low_mem=False):
    """Save the qnet to a file.

    NOTE: The file name must end in `.joblib`

    TODO: using joblib is actually less memory efficient than using pickle.
    However, I don't know if this is a general problem or this only happens
    under certain circumstances.

    TODO: we may have to delete and garbage collection some attributes in the qnet
    to save memory. For example, `.feature_importances_`, `.available_features_`

    Parameters
    ----------
    qnet1 : Qnet
        A Qnet instance

    f : str
        File name

    low_mem : bool
        If True, save the Qnet with low memory by deleting all data attributes 
        except the tree structure
 
    Returns
    -------
    None
    """

    assert isinstance(qnet, Qnet)
    if not f.endswith('.joblib'):
        raise ValueError('The outfile must end with `.joblib`')

    if low_mem:
        qnet.clear_attributes()
        
    dump(qnet, f) 


def export_qnet_trees(qnet, index, outfile, outformat='graphviz'):
    """Export the tree.

    Parameters
    ----------
    qnet : Qnet
        A Qnet instance

    index : int
        Index of the tree to export

    low_mem : bool
        If True, save the Qnet with low memory by deleting all data attributes 
        except the tree structure
 
    Returns
    -------
    None
    """

    tree = qnet.estimators_[index]
    feature_names = qnet.feature_names

    if outformat == 'graphviz':
        exporter = GraphvizExporter(tree=tree,
                         outfile=outfile,
                         response_name=feature_names[index],
                         feature_names=feature_names)
        exporter.export()

    else:
        raise NotImplementedError