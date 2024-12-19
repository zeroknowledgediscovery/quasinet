from decimal import Decimal
import threading
import time

# from numba import autojit
import numpy as np

# from externals.six.moves import range

def generate_seed():
    '''
    generates a seed as function of current time and thread id for random number generator seed.
    Must be used when large number of qsamples are drawn in parallel
    '''
    # Get the current time
    current_time = int(time.time() * 1000000)  # Convert time to microseconds for more precision
    # Extract the last 6 digits of the current time
    last_6_digits = current_time % 1000000
    # Get the current thread ID
    thread_id = threading.get_ident()
    # Combine the thread ID and the last 6 digits of the current time
    seed = (thread_id + last_6_digits) % 1000000  # Ensure the seed fits within a typical integer size
    return seed



def remove_newline_in_dotfile(file_path):
    """remove newlines from edge labels in dotfile
    """
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if '--' in line:
            line = line.replace('\\n ', '')
        new_lines.append(line)

    with open(file_path, 'w') as f:
        f.writelines(new_lines)

    return 




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

def assert_string_type(X, name):
    """Check if the input is of string datatype.

    Parameters
    ----------
    X : array-like
        Array to check

    name : str
        Name of the input
    
    Returns
    -------
    None
    """

    if not np.issubdtype(X.dtype, np.str_):
        raise ValueError('{} must contain only strings!'.format(name))

def sample_from_dict(distrib):
    """Choose an item from the distribution

    Parameters
    ----------
    distrib : dict
        Dictionary mapping keys to its probability values

    Returns
    -------
    item : key of dict
        A chosen key from the dictionary
    """

    keys = []
    probs = []
    for k, prob in distrib.items():
        keys.append(k)
        probs.append(prob)

    probs = np.array(probs)
    probs /= probs.sum()

    item = np.random.choice(keys, p=probs)

    return item


def getNull(model,strtype='U5'):
    """
    Function to generate an array of empty strings of same length as feature names in the model.

    Parameters
    ----------
    model : Qnet object
        The Qnet model.

    STRTYPE : str
        String type to be used for the generated numpy array. Default is 'U5'.

    Returns
    -------
    numpy.ndarray
        An array of empty strings.
    """
    return np.array(['']*len(model.feature_names)).astype(strtype)


def find_matching_indices(A, B):
    indices = []
    for i, value in enumerate(A):
        if value in B:
            indices.append(i)
    return indices


try:
    import pygraphviz as pgv
except ImportError:
    import warnings
    warnings.warn("pygraphviz is not installed. Some functionality may be unavailable.", ImportWarning)

import re
import os
import glob


def big_enough(dot_file,big_enough_threshold=-1):
    return len(analyze_dot_file(str(dot_file),
                                fracThreshold=.25)[1]) > big_enough_threshold


def analyze_dot_file(dot_file,fracThreshold=0.0):
    graph = pgv.AGraph(dot_file)

    non_leaf_nodes = [node for node in graph.nodes() if graph.out_degree(node) > 0]
    if len(non_leaf_nodes) <= 1:
        return False, []
    nodes_leading_to_big_frac = []

    def dfs(node):
        if graph.out_degree(node) == 0:  # if leaf node
            frac_value = re.search('Frac: ([0-9\.]+)', node.attr['label'])
            if frac_value is not None:
                return float(frac_value.group(1))
            else:
                return 0
        else:  # if non-leaf node
            frac_sum = 0
            for edge in graph.out_edges(node):
                destination_node = graph.get_node(edge[1])
                frac_sum += dfs(destination_node)
            return frac_sum

    for node in non_leaf_nodes:
        if dfs(node) > fracThreshold:
            nodes_leading_to_big_frac.append(node.attr['label'])

    return True, nodes_leading_to_big_frac

def drawtrees(dotfiles,prog='dot',format='pdf',big_enough_threshold=-1):
   for dot_file in dotfiles:
        if big_enough(dot_file,big_enough_threshold):
            graph = pgv.AGraph(str(dot_file))
            graph.draw(dot_file.replace('dot',format),
                   prog=prog, format=format) 
    
from .tree import get_nodes

def numparameters(qnetmodel):
    '''
    computes total number of prameters in qnet
    
    Parameters
    ----------
    model : Qnet object
        The Qnet model.

    Returns
    -------
    int
        number of independent parameters.
    float
        number of internal nodes per model column.

    '''
    
    leaves_all = list()
    for tree in qnetmodel.estimators_.values():
        leaves_all.append(get_nodes(tree.root, get_non_leaves=False))

    N=0
    for leaves in leaves_all:
        for leaf_distr in leaves:
            N=N+(len(leaf_distr.value)) # -1 for prob, and +1 for frac

    leaves_all_ = list()
    for tree in qnetmodel.estimators_.values():
        leaves_all_.append(get_nodes(tree.root, get_non_leaves=True))

    # number of internal nodes
    M=np.sum([len(x) for x in leaves_all_]) - np.sum([len(x) for x in leaves_all])
    # each internal node has 3 parameters (the node label and two edge labels)
    N=N+3*M  
    return N, M/len(qnetmodel.feature_names)
