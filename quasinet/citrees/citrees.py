from __future__ import absolute_import, division, print_function

import multiprocessing
import threading
import warnings
from collections import Counter

from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder

warnings.simplefilter('ignore')

# Package imports
from feature_selectors import (permutation_test_mc, permutation_test_mi,
                               permutation_test_dcor, permutation_test_pcor,
                               permutation_test_rdc, permutation_test_chi2)
from feature_selectors import mc_fast, mi, pcor, py_dcor
from scorers import gini_index, mse
from utils import bayes_boot_probs, logger, powerset


###################
"""SINGLE MODELS"""
###################


class Node(object):
    """Decision node in tree

    Parameters
    ----------
    col : int
        Integer indexing the location of feature or column

    col_pval : float
        Probability value from permutation test for feature selection

    threshold : float
        Best split found in feature

    impurity : float
        Impurity measuring quality of split

    value : 1d array-like or float
        For classification trees, estimate of each class probability
        For regression trees, central tendency estimate

    left_child : Node
        Another Node

    right_child : Node
        Another Node

    label_frequency: dict
        Dictionary mapping label to its frequency
    """

    def __init__(self, 
                 col=None, 
                 col_pval=None, 
                 threshold=None, 
                 impurity=None,
                 value=None, 
                 left_child=None, 
                 right_child=None, 
                 label_frequency=None):

        assert isinstance(left_child, Node) or left_child is None
        assert isinstance(right_child, Node) or right_child is None

        self.col         = col
        self.col_pval    = col_pval
        self.threshold   = threshold
        self.impurity    = impurity
        self.value       = value
        self.left_child  = left_child
        self.right_child = right_child
        self.label_frequency = label_frequency


# def traverse_tree(col, attribute, col_to_attribute):

class CITreeBase(object):
    """Base class for conditional inference tree

    Parameters
    ----------
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

    n_jobs : int
        Number of jobs for permutation testing

    random_state : int
        Sets seed for random number generator
    """
    def __init__(self, min_samples_split=2, alpha=.05, max_depth=-1,
                 max_feats=-1, n_permutations=100, early_stopping=False,
                 muting=True, verbose=0, n_jobs=-1, random_state=None):

        # Error checking
        if alpha <= 0 or alpha > 1:
            raise ValueError("Alpha (%.2f) should be in (0, 1]" % alpha)
        if n_permutations < 0:
            raise ValueError("n_permutations (%d) should be > 0" % \
                             n_permutations)
        if not isinstance(max_feats, int) and max_feats not in ['sqrt', 'log', 'all', -1]:
            raise ValueError("%s not a valid argument for max_feats" % \
                             str(max_feats))

        # Define attributes
        self.alpha             = float(alpha)
        self.min_samples_split = max(1, int(min_samples_split))
        self.n_permutations    = int(n_permutations)
        self.max_feats         = max_feats
        self.early_stopping    = early_stopping
        self.muting            = muting
        self.verbose           = verbose
        self.n_jobs            = n_jobs
        self.root              = None
        self.splitter_counter_ = 0

        if max_depth == -1:
            self.max_depth = np.inf
        else:
            self.max_depth = int(max(1, max_depth))

        if random_state is None:
            self.random_state = np.random.randint(1, 9999)
        else:
            # TODO: ADD CHECK FOR CRAZY LARGE INTEGER?
            self.random_state = int(random_state)


    def _mute_feature(self, col_to_mute):
        """Removes variable from being selected

        Parameters
        ----------
        col_to_mute : int
            Integer index of column to remove
        """
        # Remove feature from protected features array
        idx = np.where(self.available_features_ == col_to_mute)[0]

        # Mute feature if not in protected set
        if idx in self.protected_features_:
            return
        else:
            self.available_features_ = np.delete(self.available_features_, idx)

            # Recalculate actual number for max_feats before fitting
            p = self.available_features_.shape[0]
            if self.max_feats == 'sqrt':
                self.max_feats = int(np.sqrt(p))
            elif self.max_feats == 'log':
                self.max_feats = int(np.log(p+1))
            elif self.max_feats in ['all', -1]:
                self.max_feats = p
            else:
                self.max_feats = int(self.max_feats)

            # Check to make sure max_feats is not larger than the number of remaining
            # features
            if self.max_feats > len(self.available_features_):
                self.max_feats = len(self.available_features_)


    def _selector(self, X, y, col_idx):
        """Find feature most correlated with label"""
        raise NotImplementedError("_splitter method not callable from base class")


    def _splitter(self, *args, **kwargs):
        """Finds best split for feature"""
        raise NotImplementedError("_splitter method not callable from base class")


    def _build_tree(self, X, y, depth=0):
        """Recursively builds tree

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        depth : int
            Depth of current recursive call

        Returns
        -------
        Node : object
            Child node or terminal node in recursive splitting
        """
        n, p = X.shape

        # Check for stopping criteria
        if n > self.min_samples_split and \
           depth < self.max_depth and \
           not np.all(y == y[0]):

            # Controls randomness of column sampling
            self.splitter_counter_ += 1
            np.random.seed(self.random_state*self.splitter_counter_)

            # Find column with strongest association with outcome
            try:
                col_idx = np.random.choice(self.available_features_,
                                           size=self.max_feats, replace=False)
            except:
                col_idx = np.random.choice(self.available_features_,
                                           size=len(self.available_features_),
                                           replace=False)

            col, col_pval = self._selector(X, y, col_idx)
            
            # Add selected feature to protected features
            if col not in self.protected_features_:
                self.protected_features_.append(col)
                if self.verbose > 1:
                    logger("tree", "Added feature %d to protected set, size "
                           "= %d" % (col, len(self.protected_features_)))

            if col_pval <= self.alpha:

                # Find best split among selected variable
                impurity, threshold, left, right = self._splitter(X, y, n, col)
                if left and right and len(left[0]) > 0 and len(right[0]) > 0:

                    # Build subtrees for the right and left branches
                    if self.verbose:
                        logger("tree", "Building left subtree with "
                                       "%d samples at depth %d" % \
                                       (len(left[0]), depth+1))
                    left_child = self._build_tree(*left, depth=depth+1)

                    if self.verbose:
                        logger("tree", "Building right subtree with "
                                       "%d samples at depth %d" % \
                                        (len(right[0]), depth+1))
                    right_child = self._build_tree(*right, depth=depth+1)

                    # Return all arguments to constructor except value
                    return Node(col=col, col_pval=col_pval, threshold=threshold,
                                left_child=left_child, right_child=right_child,
                                impurity=impurity, label_frequency=Counter(y))

        # Calculate terminal node value
        if self.verbose: logger("tree", "Root node reached at depth %d" % depth)
        value = self.node_estimate(y)

        # Terminal node, no other values to pass to constructor
        return Node(value=value, label_frequency=Counter(y))


    def fit(self, X, y=None):
        """Trains model

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        TODO: check datatypes of every column in the dataframe

        Returns
        -------
        self : CITreeBase
            Instance of CITreeBase class
        """

        if not isinstance(X, pd.core.frame.DataFrame):
            raise ValueError('X must be a pandas Dataframe!')

        if not isinstance(y, np.ndarray) or len(y.shape) != 1:
            raise ValueError('y must be a 1d numpy array!')

        if self.verbose:
            logger("tree", "Building root node with %d samples" % X.shape[0])

        # Calculate actual number for max_feats before fitting
        p = X.shape[1]
        if self.max_feats == 'sqrt':
            self.max_feats = int(np.sqrt(p))
        elif self.max_feats == 'log':
            self.max_feats = int(np.log(p+1))
        elif self.max_feats in ['all', -1]:
            self.max_feats = p
        else:
            self.max_feats = int(self.max_feats)

        # Begin recursive build
        self.protected_features_  = []
        self.available_features_  = np.arange(p, dtype=int)
        self.feature_importances_ = np.zeros(p)
        self.root                 = self._build_tree(X, y)
        sum_fi                    = np.sum(self.feature_importances_)
        if sum_fi > 0: self.feature_importances_ /= sum_fi

        return self


    def predict_label(self, X, tree=None):
        """Predicts label

        Parameters
        ----------
        X : 2d array-like
            Array of features for single sample

        tree : CITreeBase
            Trained tree

        Returns
        -------
        label : int or float
            Predicted label
        """
        # If we have a value => return value as the prediction
        if tree is None: tree = self.root
        if tree.value is not None: return tree.value

        # Determine if we will follow left or right branch
        feature_value = X.iloc[tree.col]

        threshold = tree.threshold

        if isinstance(threshold, list):
            if feature_value in tree.threshold:
                branch = tree.left_child
            else:
                branch = tree.right_child
        else:
            if feature_value <= tree.threshold:
                branch = tree.left_child
            else:
                branch = tree.right_child

        # Test subtree
        return self.predict_label(X, branch)


    def predict(self, *args, **kwargs):
        """Predicts labels on test data"""
        raise NotImplementedError("predict method not callable from base class")


    def print_tree(self, tree=None, indent=" ", child=None):
        """Prints tree structure

        Parameters
        ----------
        tree : CITreeBase
            Trained tree model

        indent : str
            Indent spacing

        child : Node
            Left or right child node
        """
        # If we're at leaf => print the label
        if not tree: tree = self.root
        if tree.value is not None: print("label:", tree.value)

        # Go deeper down the tree
        else:
            # Print splitting rule
            print("X[:,%s] %s %s " % (tree.col,
                                      '<=' if child in [None, 'left'] else '>',
                                      tree.threshold))

            # Print the left child
            print("%sL: " % (indent), end="")
            self.print_tree(tree.left_child, indent + indent, 'left')

            # Print the right
            print("%sR: " % (indent), end="")
            self.print_tree(tree.right_child, indent + indent, 'right')


class CITreeClassifier(CITreeBase, BaseEstimator, ClassifierMixin):
    """Conditional inference tree classifier

    TODO: as of now, the features can only be categorical or continuous

    Parameters
    ----------
    selector : str
        Variable selector for finding strongest association between a feature
        and the label

    Derived from CITreeBase class; see constructor for parameter definitions

    """
    def __init__(self,
                 min_samples_split=2,
                 alpha=.05,
                 selector='mc',
                 max_depth=-1,
                 max_feats=-1,
                 n_permutations=100,
                 early_stopping=False,
                 muting=True,
                 verbose=0,
                 n_jobs=-1,
                 random_state=None):
        # Define node estimate
        self.node_estimate = self._estimate_proba

        # Define selector
        # if selector not in ['mc', 'mi', 'hybrid']:
        #     raise ValueError("%s not a valid selector, valid selectors are " \
        #                      "mc, mi, and hybrid")

        self.selector = selector

        if selector != 'hybrid':
            # Wrapper correlation selector
            self._selector = self._cor_selector

            # Permutation test based on correlation measure
            if selector == 'mc':
                self._perm_test = permutation_test_mc
            elif selector == 'mi':
                self._perm_test = permutation_test_mi
            elif selector == 'chi2':
                self._perm_test = permutation_test_chi2
            else:
                raise ValueError('Not a correct selector: {}'.format(selector))

        else:
            self._perm_test = None
            self._selector  = self._hybrid_selector

        super(CITreeClassifier, self).__init__(
                    min_samples_split=min_samples_split,
                    alpha=alpha,
                    max_depth=max_depth,
                    max_feats=max_feats,
                    n_permutations=n_permutations,
                    early_stopping=early_stopping,
                    muting=muting,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    random_state=random_state)


    def _hybrid_selector(self, X, y, col_idx):
        """Selects feature most correlated with y using permutation tests with
        a hybrid of multiple correlation and mutual information measures

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        col_idx : list
            Columns of X to examine for feature selection

        Returns
        -------
        best_col : int
            Best column from feature selection. Note, if early_stopping is
            enabled then this may not be the absolute best column

        best_pval : float
            Probability value from permutation test
        """
        # Select random column from start and update
        best_col, best_pval = np.random.choice(col_idx), np.inf

        # Iterate over columns
        for col in col_idx:
            if mc_fast(X[:, col], y, self.n_classes_) >= mi(X[:, col], y):
                pval = permutation_test_mc(x=X[:, col],
                                           y=y,
                                           n_classes=self.n_classes_,
                                           B=self.n_permutations,
                                           random_state=self.random_state)
            else:
                pval = permutation_test_mi(x=X[:, col],
                                           y=y,
                                           B=self.n_permutations,
                                           random_state=self.random_state)

            # If variable muting
            if self.muting and \
               pval == 1.0 and \
               self.available_features_.shape[0] > 1:
                self._mute_feature(col)
                if self.verbose: logger("tree", "ASL = 1.0, muting feature %d" % col)

            if pval < best_pval:
                best_col, best_pval = col, pval

                # If early stopping
                if self.early_stopping and best_pval < self.alpha:
                    if self.verbose: logger("tree", "Early stopping")
                    return best_col, best_pval

        return best_col, best_pval


    def gini_index(self, Y, classes):
        """Compute gini index for an array of labels.
        """
        gini = 0
        for class_ in classes:
            p_k = (Y[Y == class_]).shape[0] / Y.shape[0]
            gini += p_k * (1 - p_k)

        return gini


    def _split_by_gini(self, X, y, datatype):
        """Given a single covariate, split the data to minimize the 
        gini coefficient.

        TODO: do randomly sampling if the number of categories is large

        Parameters
        ----------
        X: 2d array-like of shape (n, 1)

        y: 1d array-like

        datatype: str
            type of data of X
        """

        index_to_gini_values = {}
        index_to_subset = {}

        unique_X = np.sort(np.unique(X))

        # the number of sets is exponential in the number of unique X values
        # TODO: when the size is large, we are going to randomly sample 2 ** 8
        # many different subsets 
        if len(unique_X) <= 8:
            subsets = powerset(unique_X)
        else:
            raise NotImplementedError

        if datatype == 'categorical':
            for i, subset in enumerate(subsets):
                # TODO: sort subset

                # ignore the case where the list is empty or the subset
                # is the set itself
                if len(subset)  == 0:
                    continue 
                elif len(subset) == len(unique_X):
                    continue

                subset_complement = unique_X[~np.isin(unique_X, subset)]
                y1 = y[np.isin(y, subset)]
                y2 = y[np.isin(y, subset_complement)]

                gini_value = gini_index(y1, subset) + gini_index(y2, subset_complement)

                index_to_gini_values[i] = gini_value
                index_to_subset[i] = subset
        else:
            raise ValueError('Datatype must be categorical!')

        min_key = min(index_to_gini_values, key=index_to_gini_values.get)
        return index_to_subset[min_key]


    def _splitter(self, X, y, n, col):
        """Splits data set into two child nodes based on optimized weighted
        gini index

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        n : int
            Number of samples

        col : list
            Column of X to search for best split

        Returns
        -------
        best_impurity : float
            Gini index associated with best split

        best_threshold : float
            X value associated with splitting of data set into two child nodes

        left : tuple
            Left child node data consisting of two elements: (features, labels)

        right : tuple
            Right child node data consisting of two elements: (features labels)
        """
        if self.verbose > 1:
            logger("splitter", "Testing splits on feature %d" % col)

        # Initialize variables for splitting
        impurity, threshold = 0.0, None
        left, right         = None, None

        X_col = X.iloc[:, col].values
        datatype = 'categorical'
        if datatype == 'categorical':
            threshold = self._split_by_gini(X_col, y, datatype)

            idx = np.where(X.iloc[:, col].isin(threshold), 1, 0)

        elif datatype == 'continuous':
            # Call sklearn's optimized implementation of decision tree classifiers
            # to make split using Gini index
            base = DecisionTreeClassifier(
                    max_depth=1, min_samples_split=self.min_samples_split
                ).fit(X_col.reshape(-1, 1), y).tree_

            # Make split based on best threshold
            threshold        = base.threshold[0]
            idx              = np.where(X[:, col] <= threshold, 1, 0)

        else:
            raise ValueError('Not a correct datatype')
        
        X_left, y_left   = X[idx==1], y[idx==1]
        X_right, y_right = X[idx==0], y[idx==0]
        n_left, n_right  = X_left.shape[0], X_right.shape[0]

        # Skip small splits
        if n_left < self.min_samples_split or n_right < self.min_samples_split:
            return impurity, threshold, left, right

        # Calculate parent and weighted children impurities
        # if len(base.impurity) == 3:
        #     node_impurity  = base.impurity[0]
        #     left_impurity  = base.impurity[1]*(n_left/float(n))
        #     right_impurity = base.impurity[2]*(n_right/float(n))
        # else:
        #     node_impurity  = gini_index(y, self.labels_)
        #     left_impurity  = gini_index(y_left, self.labels_)*(n_left/float(n))
        #     right_impurity = gini_index(y_right, self.labels_)*(n_right/float(n))


        node_impurity  = gini_index(y, self.labels_)
        left_impurity  = gini_index(y_left, self.labels_)*(n_left/float(n))
        right_impurity = gini_index(y_right, self.labels_)*(n_right/float(n))

        # Define groups and calculate impurity decrease
        left, right = (X_left, y_left), (X_right, y_right)
        impurity    = node_impurity - (left_impurity + right_impurity)

        # Update feature importance (mean decrease impurity)
        self.feature_importances_[col] += impurity

        return impurity, threshold, left, right


    def _cor_selector(self, X, y, col_idx):
        """Selects feature most correlated with y using permutation tests with
        a correlation measure

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        col_idx : list
            Columns of X to examine for feature selection

        Returns
        -------
        best_col : int
            Best column from feature selection. Note, if early_stopping is
            enabled then this may not be the absolute best column

        best_pval : float
            Probability value from permutation test
        """
        # Select random column from start and update
        best_col, best_pval = np.random.choice(col_idx), np.inf

        y = self.y_enc.transform(y.reshape(-1, 1))

        # Iterate over columns
        for col in col_idx:
            X_col = X.iloc[:, col].values
            # Mute feature and continue since constant
            if np.all( X_col == X.iloc[0, col]) and len(self.available_features_) > 1:
                self._mute_feature(col)
                if self.verbose: logger("tree", "Constant values, muting feature %d" \
                                        % col)
                continue

            X_col = self.X_encs[col].transform(X_col.reshape(-1, 1))

            pval = self._perm_test(x=X_col,
                                   y=y,
                                   n_classes=self.n_classes_,
                                   B=self.n_permutations,
                                   random_state=self.random_state)

            # If variable muting
            if self.muting and \
               pval == 1.0 and \
               self.available_features_.shape[0] > 1:
                self._mute_feature(col)
                if self.verbose: logger("tree", "ASL = 1.0, muting feature %d" % col)

            if pval < best_pval:
                best_col, best_pval = col, pval

                # If early stopping
                if self.early_stopping and best_pval < self.alpha:
                    if self.verbose: logger("tree", "Early stopping")
                    return best_col, best_pval

        return best_col, best_pval


    def _estimate_proba(self, y):
        """Estimates class distribution in node

        Parameters
        ----------
        y : 1d array-like
            Array of labels

        Returns
        -------
        class_probs : 1d array-like
            Array of class probabilities
        """
        return np.array([np.mean(y == label) for label in self.labels_])


    def fit(self, X, y, labels=None):
        """Trains conditional inference tree classifier

        Parameters
        ----------
        X : 2d array-like
            Array of features

        y : 1d array-like
            Array of labels

        labels : 1d array-like
            Array of unique class labels

        Returns
        -------
        self : CITreeClassifier
            Instance of CITreeClassifier class
        """

        if labels is None:
            labels = 'auto'
        self.y_enc = OrdinalEncoder(categories=labels, dtype=np.int32)

        self.y_enc.fit(y.reshape(-1, 1))

        self.X_encs = {}
        for col_index in np.arange(X.shape[1]):
            enc = OrdinalEncoder(dtype=np.int32)
            enc.fit(X.iloc[:, col_index].values.reshape(-1, 1))
            self.X_encs[col_index] = enc

        self.labels_ = self.y_enc.categories_[0]
        self.n_classes_ = len(self.labels_)
        super(CITreeClassifier, self).fit(X, y)
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

        num_samples = X.shape[0]
        predictions = np.zeros((num_samples, self.n_classes_))
        for i in np.arange(0, num_samples):
            predictions[i] = self.predict_label(X.iloc[i])

        return predictions
        # return np.array([self.predict_label(sample) for sample in X])


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
        ordinal_prediction = np.argmax(y_proba, axis=1).reshape((-1, 1))
        categorical_prediction = self.y_enc.inverse_transform(ordinal_prediction)

        return categorical_prediction.reshape(-1)
        



