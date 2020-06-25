#!/usr/bin/env python
# coding: utf-8

# # Parameters to Set

# In[1]:


NUMCPUS = 5

# MAX_COLS = 20
MAX_COLS = None

# MAX_ROWS = 10
MAX_ROWS = None

TRAIN_QNETS = True

COMPUTE_QDISTS = True


# ## Paths

# In[2]:


# DATA_DIR = '/home/jinli11/quasinet/data/influenza/trees/h1n1humanHA/'
DATA_DIR = '/project2/ishanu/hiv-dip/influenza/trees/h1n1humanHA/'

OUTPUT_DIR = 'output/'

INFLUENZA_OUT_DIR = OUTPUT_DIR + 'influenza/'

INFLUENZA_QNET_DIR = INFLUENZA_OUT_DIR + 'qnets/'

INFLUENZA_QDIST_DIR = INFLUENZA_OUT_DIR + 'qdistances/'


# # Imports

# In[3]:


from importlib import reload
import sys
import glob
import os
import pickle
from joblib import Parallel, delayed

import sklearn
import pandas as pd
from sklearn.datasets import load_iris, load_wine

# sys.path.insert(1, '/home/jinli11/quasinet/quasinet/citrees/')
sys.path.insert(1, '/project2/ishanu/hiv-dip/quasinet/quasinet/citrees/')

# import citrees
import qnet
import tree

# reload(qnet)
# reload(citrees)


# # Notes

# * `jupyter nbconvert --to script influenza.ipynb`

# # Helper Functions
# 

# In[4]:


def load_csv_files(dir_):
    f_to_data = {}
    for f in glob.glob(dir_ + '*.csv'):
        f_to_data[os.path.basename(f)] = pd.read_csv(f)
        
    return f_to_data

    
def make_dir(dir_):
    """Make a directory if it doesn't exist.

    Args:
        dir (str): directory to make
    """
    
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
        
def load_pickled(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def save_pickled(item, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)


# # Making Directories

# In[5]:


make_dir(INFLUENZA_OUT_DIR)
make_dir(INFLUENZA_QNET_DIR)
make_dir(INFLUENZA_QDIST_DIR)


# # Loading

# In[6]:


f_to_seqs = load_csv_files(DATA_DIR)


# In[7]:


# list(f_to_seqs.values())[3].shape


# # Qnet Training

# ## Functions

# In[8]:


def train_qnet(f, seqs, output_dir, max_cols, numCPUs):
    seqs = seqs.values.astype(str)[:, :max_cols]
    myqnet = qnet.Qnet(n_jobs=numCPUs)
    myqnet.fit(seqs)

    basename = f.replace('.csv', '.joblib')
    if output_dir is not None:
        outfile = os.path.join(output_dir, basename)
        qnet.save_qnet(myqnet, outfile)
        
    return basename, myqnet
            
def train_qnets(f_to_seqs, numCPUs, output_dir, max_cols=None):
        
    keys = list(f_to_seqs.keys())#[:2]
    
    base_name_trees = Parallel(n_jobs=numCPUs, backend='loky')(
        delayed(train_qnet)(
            key, f_to_seqs[key], output_dir, max_cols, numCPUs)
        for key in keys
        )
    
    f_to_qnet = {basename: tree for basename, tree in base_name_trees}
    return f_to_qnet


# ## Qnet Training

# In[9]:


# result = train_qnet(
#     'h1n1human2009_2010.csv',
#     f_to_seqs['h1n1human2009_2010.csv'], 
#     output_dir=None, 
#     max_cols=MAX_COLS, 
#     numCPUs=1)


# In[10]:


if TRAIN_QNETS:
    f_to_qnets = train_qnets(
        f_to_seqs, 
        numCPUs=NUMCPUS, 
        output_dir=INFLUENZA_QNET_DIR, 
        max_cols=MAX_COLS)


# # QDistance Computation

# ## Functions

# In[11]:


def load_qnets(dir_):
    f_to_qnets = {}
    for f in glob.glob(dir_ + "*.joblib"):
        f_to_qnets[os.path.basename(f)] = qnet.load_qnet(f)
        
    return f_to_qnets

def compute_qdist(f, seqs, myqnet, max_seqs, max_cols):
    seqs = seqs.drop_duplicates()
    indices = seqs.index[:max_seqs]
    seqs = seqs.values

    dm = qnet.qdistance_matrix(
        seqs[:max_seqs, :max_cols], 
        seqs[:max_seqs, :max_cols],
        myqnet, 
        myqnet)

    dm = pd.DataFrame(dm, index=indices, columns=indices)

    return f, dm
    
def compute_qdists(f_to_qnets, f_to_seqs, max_seqs, max_cols, numCPUs, outdir):
    """"""
    
    keys = list(f_to_seqs.keys())#[:2]
    qnet_names = [key.split('.')[0] + '.joblib' for key in keys]
    
    f_dms = Parallel(n_jobs=numCPUs, backend='loky')(
        delayed(compute_qdist)(
            key, f_to_seqs[key], f_to_qnets[qnet_names[i]], max_seqs, max_cols)
        for i, key in enumerate(keys)
        )
    
    f_to_dms = {}
    for f, dm in f_dms:
        f_to_dms[f] = dm
        dm.to_csv(os.path.join(outdir, f))
#     f_to_dms = {f: dm for f, dm in f_dms}
    
    return f_to_dms


# ## Computation

# In[12]:


# f_to_qnets = load_qnets(INFLUENZA_QNET_DIR)


# In[13]:


# sys.getsizeof(f_to_qnets['h1n1human2000_2001.joblib'].estimators_) #.estimators_[1].root
# tree.get_nodes(f_to_qnets['h1n1human2000_2001.joblib'].estimators_[2].root)
# save_pickled(f_to_qnets['h1n1human2000_2001.joblib'], 'TMP.pkl')
# save_pickled(f_to_qnets['h1n1human2000_2001.joblib'].estimators_, 'TMP2.pkl')
# save_pickled(f_to_qnets['h1n1human2000_2001.joblib'].estimators_[0], 'TMP3.pkl')
# f_to_qnets['h1n1human2000_2001.joblib'].estimators_[0].feature_importances_.nbytes


# In[14]:


if COMPUTE_QDISTS:
    qdist = compute_qdists(
        f_to_qnets, f_to_seqs, max_seqs=MAX_ROWS, 
        max_cols=MAX_COLS, numCPUs=NUMCPUS ** 2, outdir=INFLUENZA_QDIST_DIR)


# In[15]:


# qdist


# In[ ]:




