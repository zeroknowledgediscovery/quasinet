
import pandas as pd
import numpy as np
from joblib import dump, load

from citrees import CITreeClassifier
from metrics import js_divergence

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
        for col in X.columns:
            clf = CITreeClassifier(selector='chi2')
            y = X[col].values
            clf.fit(X.drop([col], axis=1), y)
            self.estimators_[col] = clf


    def check_is_fitted(self):
        if not hasattr(self, 'estimators_'):
            raise ValueError('You need to call `fit` first! ')


    def predict_distribution(self, seq, column):
        """Predict the probability distribution for a given column.

        Parameters
        ----------
        seq : list
            list of items in the sequence

        column : str
            column name

        Returns
        -------
        output : dictionary
            dictionary mapping possible column values to probability values
        """

        self.check_is_fitted()

        raise NotImplementedError


    def predict_distributions(self, seq):
        """Predict the probability distributions for all the columns.

        Parameters
        ----------
        seq : list
            list of items in the sequence

        Returns
        -------
        output : dictionary
            dictionary mapping column names to another dictionary. That dictionary
            maps possible column values to probability values
        """

        self.check_is_fitted()

        col_to_prob_distrib = {}
        for col in self.estimators_.keys():
            col_to_prob_distrib[col] = self.predict_distribution(seq, col)

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

    pass


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