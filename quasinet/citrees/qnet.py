
import pandas as pd
import numpy as np

from citrees import CITreeClassifier

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

    def predict_proba(self, X):
        pass
