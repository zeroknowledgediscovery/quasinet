import multiprocessing
import threading
import warnings

from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder

warnings.simplefilter('ignore')

# Package imports
from .feature_selectors import (permutation_test_mc, permutation_test_mi,
                               permutation_test_dcor, permutation_test_pcor,
                               permutation_test_rdc, permutation_test_chi2)
from .feature_selectors import mc_fast, mi, pcor, py_dcor
from .scorers import gini_index, mse
from .utils import bayes_boot_probs, logger, powerset
from .citrees import CITreeClassifier



def stratify_sampled_idx(random_state, y, bayes):
    """Indices for stratified bootstrap sampling in classification

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    y : 1d array-like
        Array of labels

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    Returns
    -------
    idx : list
        Stratified sampled indices for each class
    """
    np.random.seed(random_state)
    idx = []
    for label in np.unique(y):

        # Grab indices for class
        tmp = np.where(y==label)[0]

        # Bayesian bootstrapping if specified
        p = bayes_boot_probs(n=len(tmp)) if bayes else None

        idx.append(np.random.choice(tmp, size=len(tmp), replace=True, p=p))

    return idx


def stratify_unsampled_idx(random_state, y, bayes):
    """Unsampled indices for stratified bootstrap sampling in classification

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    y : 1d array-like
        Array of labels

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    Returns
    -------
    idx : list
        Stratified unsampled indices for each class
    """
    np.random.seed(random_state)
    sampled = stratify_sampled_idx(random_state, y, bayes)
    idx     = []
    for i, label in enumerate(np.unique(y)):
        idx.append(np.setdiff1d(np.where(y==label)[0], sampled[i]))
    return idx


def balanced_sampled_idx(random_state, y, bayes, min_class_p):
    """Indices for balanced bootstrap sampling in classification

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    y : 1d array-like
        Array of labels

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    min_class_p : float
        Minimum proportion of class labels

    Returns
    -------
    idx : list
        Balanced sampled indices for each class
    """
    np.random.seed(random_state)
    idx, n = [], int(np.floor(min_class_p*len(y)))
    for i, label in enumerate(np.unique(y)):

        # Grab indices for class
        tmp = np.where(y==label)[0]

        # Bayesian bootstrapping if specified
        p = bayes_boot_probs(n=len(tmp)) if bayes else None

        idx.append(np.random.choice(tmp, size=n, replace=True, p=p))

    return idx


def balanced_unsampled_idx(random_state, y, bayes, min_class_p):
    """Unsampled indices for balanced bootstrap sampling in classification

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    y : 1d array-like
        Array of labels

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    min_class_p : float
        Minimum proportion of class labels

    Returns
    -------
    idx : list
        Balanced unsampled indices for each class
    """
    np.random.seed(random_state)
    sampled = balanced_sampled_idx(random_state, y, bayes, min_class_p)
    idx     = []
    for i, label in enumerate(np.unique(y)):
        idx.append(np.setdiff1d(np.where(y==label)[0], sampled[i]))
    return idx


def normal_sampled_idx(random_state, n, bayes):
    """Indices for bootstrap sampling

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    n : int
        Sample size

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    Returns
    -------
    idx : list
        Sampled indices
    """
    np.random.seed(random_state)

    # Bayesian bootstrapping if specified
    p = bayes_boot_probs(n=n) if bayes else None

    return np.random.choice(np.arange(n, dtype=int), size=n, replace=True, p=p)


def normal_unsampled_idx(random_state, n, bayes):
    """Unsampled indices for bootstrap sampling

    Parameters
    ----------
    random_state : int
        Sets seed for random number generator

    y : 1d array-like
        Array of labels

    n : int
        Sample size

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    Returns
    -------
    idx : list
        Unsampled indices
    """
    sampled = normal_sampled_idx(random_state, n, bayes)
    counts  = np.bincount(sampled, minlength=n)
    return np.arange(n, dtype=int)[counts==0]


def _parallel_fit_classifier(tree, X, y, n, tree_idx, n_estimators, bootstrap,
                             bayes, verbose, random_state, class_weight=None,
                             min_dist_p=None):
    """Utility function for building trees in parallel

    Note: This function can't go locally in a class, because joblib complains
          that it cannot pickle it when placed there

    Parameters
    ----------
    tree : CITreeClassifier
        Instantiated conditional inference tree

    X : 2d array-like
        Array of features

    y : 1d array-like
        Array of labels

    n : int
        Number of samples

    tree_idx : int
        Index of tree in forest

    n_estimators : int
        Number of total estimators

    bootstrap : bool
        Whether to perform bootstrap sampling

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    verbose : bool or int
        Controls verbosity of training process

    random_state : int
        Sets seed for random number generator

    class_weight : str
        Type of sampling during bootstrap, None for regular bootstrapping,
        'balanced' for balanced bootstrap sampling, and 'stratify' for
        stratified bootstrap sampling

    min_class_p : float
        Minimum proportion of class labels

    Returns
    -------
    tree : CITreeClassifier
        Fitted conditional inference tree
    """
    # Print status if conditions met
    if verbose and n_estimators >= 10:
        denom = n_estimators if verbose > 1 else 10
        if (tree_idx+1) % int(n_estimators/denom) == 0:
            logger("tree", "Building tree %d/%d" % (tree_idx+1, n_estimators))

    # Bootstrap sample if specified
    if bootstrap:
        random_state = random_state*(tree_idx+1)
        if class_weight == 'balanced':
            idx = np.concatenate(
                balanced_sampled_idx(random_state, y, bayes, min_dist_p)
                )
        elif class_weight == 'stratify':
            idx = np.concatenate(
                stratify_sampled_idx(random_state, y, bayes)
                )
        else:
            idx = normal_sampled_idx(random_state, n, bayes)

        # Note: We need to pass the classes in the case of the bootstrap
        # because not all classes may be sampled and when it comes to prediction,
        # the tree models learns a different number of classes across different
        # bootstrap samples
        tree.fit(X[idx], y[idx], np.unique(y))
    else:
        tree.fit(X, y)
    
    return tree



def _accumulate_prediction(predict, X, out, lock):
    """Utility function to aggregate predictions in parallel

    Parameters
    ----------
    predict : function handle
        Alias to prediction method of class

    X : 2d array-like
        Array of features

    out : 1d or 2d array-like
        Array of labels

    lock : threading lock
        A lock that controls worker access to data structures for aggregating
        predictions

    Returns
    -------
    None
    """
    prediction = predict(X)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            # TODO: need to check this because we used the range from external
            # need to check all otther ranges as well
            for i in range(len(out)): out[i] += prediction[i]


class CIForestClassifier(BaseEstimator, ClassifierMixin):
    """Conditional forest classifier

    Parameters
    ----------
    min_samples_split : int
        Minimum samples required for a split

    alpha : float
        Threshold value for selecting feature with permutation tests. Smaller
        values correspond to shallower trees

    selector : str
        Variable selector for finding strongest association between a feature
        and the label

    max_depth : int
        Maximum depth to grow tree

    max_feats : str or int
        Maximum feats to select at each split. String arguments include 'sqrt',
        'log', and 'all'

    n_permutations : int
        Number of permutations during feature selection

    early_stopping : bool
        Whether to implement early stopping during feature selection. If True,
        then as soon as the first permutation test returns a p-value less than
        alpha, this feature will be chosen as the splitting variable

    muting : bool
        Whether to perform variable muting

    verbose : bool or int
        Controls verbosity of training and testing

    bootstrap : bool
        Whether to perform bootstrap sampling for each tree

    bayes : bool
        If True, performs Bayesian bootstrap sampling

    class_weight : str
        Type of sampling during bootstrap, None for regular bootstrapping,
        'balanced' for balanced bootstrap sampling, and 'stratify' for
        stratified bootstrap sampling

    n_jobs : int
        Number of jobs for permutation testing

    random_state : int
        Sets seed for random number generator
    """
    def __init__(self, min_samples_split=2, alpha=.05, selector='mc', max_depth=-1,
                 n_estimators=100, max_feats='sqrt', n_permutations=100,
                 early_stopping=True, muting=True, verbose=0, bootstrap=True,
                 bayes=True, class_weight='balanced', n_jobs=-1, random_state=None):

        # Error checking
        if alpha <= 0 or alpha > 1:
            raise ValueError("Alpha (%.2f) should be in (0, 1]" % alpha)
        if selector not in ['mc', 'mi', 'hybrid']:
            raise ValueError("%s not a valid selector, valid selectors are " \
                             "mc, mi, and hybrid")
        if n_permutations < 0:
            raise ValueError("n_permutations (%s) should be > 0" % \
                             str(n_permutations))
        if not isinstance(max_feats, int) and max_feats not in ['sqrt', 'log', 'all', -1]:
            raise ValueError("%s not a valid argument for max_feats" % \
                             str(max_feats))
        if n_estimators < 0:
            raise ValueError("n_estimators (%s) must be > 0" % \
                             str(n_estimators))

        # Only for classifier model
        if class_weight not in [None, 'balanced', 'stratify']:
            raise ValueError("%s not a valid argument for class_weight" % \
                             str(class_weight))

        # Placeholder variable for regression model (not applicable)
        if class_weight is None: self.min_class_p = None

        # Define attributes
        self.alpha             = float(alpha)
        self.selector          = selector
        self.min_samples_split = max(1, min_samples_split)
        self.n_permutations    = int(n_permutations)
        if max_depth == -1:
            self.max_depth = max_depth
        else:
            self.max_depth = int(max(1, max_depth))

        self.n_estimators   = int(max(1, n_estimators))
        self.max_feats      = max_feats
        self.bootstrap      = bootstrap
        self.early_stopping = early_stopping
        self.muting         = muting
        self.n_jobs         = n_jobs
        self.verbose        = verbose
        self.class_weight   = class_weight
        self.bayes          = bayes

        if random_state is None:
            self.random_state = np.random.randint(1, 9999)
        else:
            # TODO: ADD CHECK FOR CRAZY LARGE INTEGER?
            self.random_state = int(random_state)

        # Package params for calling CITreeClassifier
        self.params = {
            'alpha'             : self.alpha,
            'selector'          : self.selector,
            'min_samples_split' : self.min_samples_split,
            'n_permutations'    : self.n_permutations,
            'max_feats'         : self.max_feats,
            'early_stopping'    : self.early_stopping,
            'muting'            : self.muting,
            'verbose'           : 0,
            'n_jobs'            : 1,
            'random_state'      : None,
            }


    def fit(self, X, y):
        """Fit conditional forest classifier

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        Returns
        -------
        self : CIForestClassifier
            Instance of CIForestClassifier
        """
        self.labels_    = np.unique(y)
        self.n_classes_ = len(self.labels_)

        if self.verbose:
            logger("tree", "Training ensemble with %d trees on %d samples" % \
                    (self.n_estimators, X.shape[0]))

        # Instantiate base tree models
        self.estimators_ = []
        for i in range(self.n_estimators):
            self.params['random_state'] = self.random_state*(i+1)
            self.estimators_.append(CITreeClassifier(**self.params))

        # Define class distribution
        self.class_dist_p = np.array([
                np.mean(y==label) for label in np.unique(y)
            ])

        # Train models
        n = X.shape[0]
        self.estimators_ = \
            Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(_parallel_fit_classifier)(
                self.estimators_[i], X, y, n, i, self.n_estimators,
                self.bootstrap, self.bayes, self.verbose, self.random_state,
                self.class_weight, np.min(self.class_dist_p)
                )
            for i in range(self.n_estimators)
            )

        # Accumulate feature importances (mean decrease impurity)
        self.feature_importances_ = np.sum([
                tree.feature_importances_ for tree in self.estimators_],
                axis=0
            )
        sum_fi = np.sum(self.feature_importances_)
        if sum_fi > 0: self.feature_importances_ /= sum_fi

        return self


    def predict_proba(self, X):
        """Predicts class probabilities for feature vectors X

        Parameters
        ----------
        X : 2d array-like
            Array of features

        Returns
        -------
        class_probs : 2d array-like
            Array of predicted class probabilities
        """
        if self.verbose:
            logger("test", "Predicting labels for %d samples" % X.shape[0])

        # Parallel prediction
        all_proba = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        lock      = threading.Lock()
        Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_accumulate_prediction)(e.predict_proba, X, all_proba, lock)
            for e in self.estimators_)

        # Normalize probabilities
        all_proba /= len(self.estimators_)
        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba


    def predict(self, X):
        """Predicts class labels for feature vectors X

        Parameters
        ----------
        X : 2d array-like
            Array of features

        Returns
        -------
        y : 1d array-like
            Array of predicted classes
        """
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)