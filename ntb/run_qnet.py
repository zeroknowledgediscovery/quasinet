from importlib import reload
import sys

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
# y = X['0'].values
# X.drop(['0'], axis=1, inplace=True)
X.columns = np.arange(0, X.shape[1])
X = X.iloc[:, :10]

myqnet = qnet.Qnet()
myqnet.fit(X)

file_ = OUTPUT_DIR + 'saved_qnet.joblib'

myqnet.estimators_[0].print_tree()
# myqnet.predict_distribution({5: 'C', 3: 'G', 2: 'A'}, 0)

seq1 = list(X[1].values)
seq2 = list(X[2].values)
# myqnet.predict_distributions(seq1)
# qnet.save_qnet(myqnet, file_)
qdist = qnet.qdistance(seq1, seq2, myqnet, myqnet) 
# myqnet = qnet.load_qnet(file_)
import pdb; pdb.set_trace()