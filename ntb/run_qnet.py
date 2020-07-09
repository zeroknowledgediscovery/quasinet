from importlib import reload
import sys
import time
import copy

import sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine

sys.path.insert(1, '/home/jinli11/quasinet/quasinet/citrees/')

import citrees
import qnet


DATA_DIR = 'data/'
OUTPUT_DIR = 'output/'

X = pd.read_csv(DATA_DIR + 'cchfl_test.csv')
# X = copy.deepcopy(X_orig)
# y = X['0'].values
# X.drop(['0'], axis=1, inplace=True)
# X.columns = np.arange(0, X.shape[1])
X = X.iloc[:, :10]
X = X.values.astype(str)
# breakpoint()
myqnet = qnet.Qnet(n_jobs=4)
myqnet.fit(X)

file_ = OUTPUT_DIR + 'saved_qnet.joblib'

# myqnet.estimators_[0].print_tree()
# myqnet.predict_distribution({5: 'C', 3: 'G', 2: 'A'}, 0)

seq1 = X[1]
seq2 = X[2]
# myqnet.predict_distributions(seq1)
# qnet.save_qnet(myqnet, file_)
# qdist = qnet.qdistance(seq1, seq1, myqnet, myqnet) 
# print (qdist)
# breakpoint()
max_seq = 20

start = time.time()
qdist_matrix = qnet.qdistance_matrix(X[:max_seq], X[:max_seq], myqnet, myqnet)

end = time.time()
print(end - start)

qnet_membership = qnet.membership_degree(seq1, myqnet)
# myqnet = qnet.load_qnet(file_)
# breakpoint()