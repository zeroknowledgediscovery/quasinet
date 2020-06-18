
import pandas as pd
import numpy as np
from joblib import dump, load

from citrees import CITreeClassifier
from metrics import js_divergence
from tree import Node, get_nodes

class Qnet(object):
    """
    """

    def __init__(self):
        pass

    def fit(self, X):

        if not isinstance(X, pd.core.frame.DataFrame):
            raise ValueError('X must be a pandas Dataframe!')

        # Instantiate base tree models
        self.estimators_ = {}

        # TODO: allow for multiple threads for speedup
        # TODO: allow for more arguments to be passed to CITrees
        # TODO: we may not have any trees created. When that's the
        # case, we want to predict an equal probability distribution
        for col in X.columns:
            clf = CITreeClassifier(selector='chi2')
            y = X[col].values
            clf.fit(X.drop([col], axis=1), y)
            self.estimators_[col] = clf


    def check_is_fitted(self):
        if not hasattr(self, 'estimators_'):
            raise ValueError('You need to call `fit` first! ')


    def predict_distribution(self, column_to_item, column):
        """Predict the probability distribution for a given column.

        TODO: check the case where `column_to_item` is empty

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

        self.check_is_fitted()

        if len(column_to_item) == 0:
            raise NotImplementedError

        root = self.estimators_[column].root
        nodes = get_nodes(root)

        distributions = {}
        for node in nodes:
            if node.col in column_to_item:
                if column_to_item[column] in node.threshold:
                    next_node = node.left
                else:
                    next_node = node.right

                distributions[column] = next_node.label_frequency

        distributions = pd.DataFrame(distributions)
        distributions.fillna(0, inplace=True)

        total_frequency = distributions.sum()
        distributions *= distributions.sum(axis=1)
        distributions /= total_frequency

        distributions = distributions.mean(axis=0)
        distributions /= distributions.sum()

        return distributions[0].to_dict()



    def predict_distributions(self, column_to_item):
        """Predict the probability distributions for all the columns.

        Parameters
        ----------
        column_to_item : dict
            dictionary mapping the column to the values the columns take

        Returns
        -------
        output : dictionary
            dictionary mapping column names to another dictionary. That dictionary
            maps possible column values to probability values.
        """

        self.check_is_fitted()

        col_to_prob_distrib = {}
        for col in self.estimators_.keys():
            col_to_prob_distrib[col] = self.predict_distribution(column_to_item, col)

        return col_to_prob_distrib


    def predict_proba(self, X):

        self.check_is_fitted()

        raise NotImplementedError



def qdistance(seq1, seq2, qnet1, qnet2):
    """Compute the Jensen-Shannon of discrete probability distributions.

    Parameters
    ----------
    seq1 : list
        list of items in sequence 1

    seq2 : list
        list of items in sequence 2

    qnet1 : Qnet
        the Qnet that `seq1` belongs to 

    qnet2 : Qnet
        the Qnet that `seq2` belongs to 

    Returns
    -------
    output : numeric
        qdistance
    """

    if len(seq1) != len(seq2):
        raise ValueError('The two sequences must be of equal lengths.')

    if (not isinstance(seq1, list)) or (not isinstance(seq2, list)):
        raise ValueError('You must pass in lists as sequences.')

    seq1_distribs = qnet1.predict_distributions(seq1)
    seq2_distribs = qnet2.predict_distributions(seq2)

    total_divergence = 0
    for col, seq1_distrib in seq1_distribs.items():

        seq2_distrib = seq2_distribs[col]

        # make sure both distribution contains the same responses
        for seq1_response in seq1_distrib.keys():
            if seq1_response not in seq2_distrib:
                seq2_distrib[seq1_response] = 0.0

        for seq2_response in seq2_distrib.keys():
            if seq2_response not in seq1_distrib:
                seq1_distrib[seq2_response] = 0.0

        num_responses = len(seq2_distrib)
        distrib1 = np.empty(num_responses)
        distrib2 = np.empty(num_responses)
        for i, x in enumerate(seq1_distrib.keys()):
            distrib1[i] = seq1_distrib[x]
            distrib2[i] = seq2_distrib[x]

        total_divergence += np.sqrt(js_divergence(distrib1, distrib2))

    total_divergence /= len(seq1_distribs)
    
    return total_divergence


def qdistance_with_prob_distributions(distrib1, distrib2):
    raise NotImplementedError

def qdistance_matrix():
    raise NotImplementedError

def load_qnet(f):
    return load(f)

def save_qnet(qnet, f):
    assert isinstance(qnet, Qnet)
    if not f.endswith('.joblib'):
        raise ValueError('The outfile must end with `.joblib`')

    dump(qnet, f) 