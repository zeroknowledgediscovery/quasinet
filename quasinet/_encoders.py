import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from ._config import get_config

class OrdinalEncoderWithNaN(OrdinalEncoder):
    """Convert string inputs into ordinal representations with custom NaN value.

    We will encode NaN values with -1.
    """

    def __init__(self, dtype):
        super(OrdinalEncoderWithNaN, self).__init__(dtype=dtype)

    def transform(self, X):

        nan_value = get_config()['nan_value']
        X = super(OrdinalEncoderWithNaN, self).transform(X)
        categories = self.categories_[0]
        if nan_value in categories:
            X[X == list(categories).index(nan_value)] = -1

        return X