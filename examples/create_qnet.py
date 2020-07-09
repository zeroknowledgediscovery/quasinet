
"""This is a basic example of how to train a qnet 
and use it to compute distances.
"""

import pandas as pd
import numpy as np

from quasinet import qnet

DATA_DIR = 'example_data/'

X = pd.read_csv(DATA_DIR + 'cchfl_test.csv')
X = X.values.astype(str)

# initialize the qnet
myqnet = qnet.Qnet(n_jobs=1)

# train the qnet
myqnet.fit(X)

# calculate qdistance
seq1 = X[1]
seq2 = X[2]
qdist = qnet.qdistance(seq1, seq2, myqnet, myqnet) 

# calculate membership degree
qnet_membership = qnet.membership_degree(seq1, myqnet)