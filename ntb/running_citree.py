#!/usr/bin/env python
# coding: utf-8

# # Data Directory

# In[2]:


DATA_DIR = 'data/'


# # Notes

# `jupyter nbconvert --to script running_citree.ipynb`

# # Imports

# In[3]:


from importlib import reload
import sys

import sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
# from scipy import special.kl_div

sys.path.insert(1, '/home/jinli11/quasinet/quasinet/citrees/')

import citrees
reload(citrees)


# In[14]:


X = pd.read_csv(DATA_DIR + 'cchfl_test.csv')
y = X['0'].values
X.drop(['0'], axis=1, inplace=True)
X = X.iloc[:, :5]
# import pdb; pdb.set_trace()
# X['dsfds'] = np.arange(0, X.shape[0])
# # y = pd.read_csv(DATA_DIR + 'abalone.names', header=None)
# y = X[['0']]
# X.drop(['0'], axis=1, inplace=True)

# x = pd.DataFrame(0, columns=['a', 'b'])
# import pdb; pdb.set_trace()

# In[16]:


# X


# In[18]:


# y


# In[11]:


# X, y = load_iris(return_X_y=True)


# In[6]:


# y.shape
# clf = citrees.CITreeClassifier(selector='chi2')


# In[7]:


# clf.fit(X, y)


# In[8]:


# pred = clf.predict(X)
# y == pred

import pdb; pdb.set_trace()


# In[ ]:




