
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

COMPUTE_COMMON_STRAIN = False

TYPE = 'h1n1'


# ## Paths

# In[2]:


# DATA_DIR = '/home/jinli11/quasinet/data/influenza/trees/h1n1humanHA/'
DATA_DIR = '/project2/ishanu/hiv-dip/influenza/trees/h1n1humanHA/'

OUTPUT_DIR = 'output/'

INFLUENZA_OUT_DIR = OUTPUT_DIR + 'influenza/'

INFLUENZA_QNET_DIR = INFLUENZA_OUT_DIR + 'qnets/'

INFLUENZA_QDIST_DIR = INFLUENZA_OUT_DIR + 'qdistances/'

HUMAN_HA_YEARLY_DIST_MATRIX_LDISTANCE_DIR = '/project2/ishanu/hiv-dip/influenza/output/yearly_distance_matrices_ldistance/h1n1humanHA/'


# # Imports

# In[3]:


from importlib import reload
import sys
import glob
import os
import re
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

import compute_common_strain
from compute_common_strain import *
reload(compute_common_strain)
# reload(citrees)


# # Notes

# * `jupyter nbconvert --to script influenza.ipynb`
# * TODO: I need to clean the sequences to reduce redudant sequences

# # Helper Functions
# 

# In[4]:


def load_csv_files(dir_, index_col=None):
    f_to_data = {}
    for file_ in glob.glob(dir_ + '*.csv'):
        f = os.path.basename(file_)
        f = re.findall(r'\d+_\d+', f)[0]
        
        f_to_data[f] = pd.read_csv(file_, index_col=index_col)
        
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
        
def remove_index_and_cols(f_to_df):
    
    new_f_to_df = {}
    for f, df in f_to_df.items():
        df = df.copy(deep=True)
        df.index = np.arange(0, df.shape[0])
        df.columns = np.arange(0, df.shape[1])
        
        new_f_to_df[f] = df
        
    return new_f_to_df


# # Making Directories

# In[5]:


make_dir(INFLUENZA_OUT_DIR)
make_dir(INFLUENZA_QNET_DIR)
make_dir(INFLUENZA_QDIST_DIR)


# # Loading

# ## Functions

# In[6]:


def year_to_cleaned_sequence_data(sequence_data):
    """Basically clean the data so we can use it for computation.
    """
    
    data = {}
    
    for year, seq_data_ in sequence_data.items():
        
        seq_data = seq_data_.copy()
        num_seqs, num_cols = seq_data.shape
        
        seq_data.drop_duplicates(inplace=True, subset=np.arange(0, num_cols).astype(str))
        seq_data.reset_index(drop=True, inplace=True)
        
        data[year] = seq_data
        
    return data


# ## Loading

# In[7]:


human_ha_seqs = load_csv_files(DATA_DIR)
human_ha_max_len = list(human_ha_seqs.values())[0].shape[1]
print ("ha max length: ", human_ha_max_len)
human_ha_cleaned_seqs = year_to_cleaned_sequence_data(human_ha_seqs)


# # Qnet Training

# ## Functions

# In[8]:


def train_qnet(f, seqs, output_dir, max_cols, numCPUs):
    seqs = seqs.values.astype(str)[:, :max_cols]
    myqnet = qnet.Qnet(n_jobs=numCPUs)
    myqnet.fit(seqs)

#     basename = f.replace('.csv', '.joblib')
    basename = f + '.joblib'
    if output_dir is not None:
        outfile = os.path.join(output_dir, basename)
#         print (outfile)
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
        human_ha_cleaned_seqs, 
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
        f_to_qnets, human_ha_seqs, max_seqs=MAX_ROWS, 
        max_cols=MAX_COLS, numCPUs=NUMCPUS ** 2, outdir=INFLUENZA_QDIST_DIR)


# In[15]:


# human_ha_qdistance_dm = load_csv_files(INFLUENZA_QDIST_DIR, index_col=0)
# human_ha_ldistance_dm = load_csv_files(HUMAN_HA_YEARLY_DIST_MATRIX_LDISTANCE_DIR, index_col=0)


# # Compute Common Strain

# ## Functions

# ### Compute Dominant Strain

# In[16]:


def compute_dominant_strain(
    name_to_dm, n_clusters, 
    upper_diag,
    file_to_seqs=None, 
    remove_outliers=False,
    clustering_method='agg',
    min_type='average'):
    """Seperate the sequences into clusters, find the largest cluster, and find the 
    centroid of that largest cluster.
    
    Args:
        name_to_dm (dict): mapping file name to distance matrix
        use_accession (bool): whether we are using the accession or not
        min_type (str): which type to use to compute the minimum
        remove_outliers (bool): whether to remove outliers. N
    """
    
    dominant_strains = []
    names = []
    seqs = []
    accession_names = []
    for name_, dm in name_to_dm.items():
        name = name_
        dm.fillna(0, inplace=True)
        names.append(name)
        columns = dm.columns.astype(str)
        index = dm.index.astype(str)
        
        #if dm.shape[0] == dm.shape[1]:
        if upper_diag:
            dm = dm.values + dm.T.values
            
        dm = pd.DataFrame(dm, columns=columns, index=index)
        
        if remove_outliers:
            dm = remove_outliers_func(dm)
            
        if min_type in ['average', 'median']:
            if n_clusters == 1:
                sub_dm = dm
            else:
                clusters = find_clusters(
                    dm, n_clusters=n_clusters, 
                    cluster_type=clustering_method)
                
                sub_dm = find_largest_cluster_dm(dm, clusters)

            if dm.shape == (1, 1):
                dominant_strain = dm.index[0]
            else:
                if min_type == 'average':
                    aggregated = sub_dm.sum(axis=1)
                elif min_type == 'median':
                    aggregated = sub_dm.median(axis=1)
                else:
                    raise ValueError
                    
                dominant_strain = aggregated.idxmin()
                #dominant_strain = np.argsort(aggregated)[len(aggregated)//2]
                
        elif min_type == 'normal':
            
            embedding = MDS(
                n_components=1, 
                dissimilarity="precomputed", 
                random_state=42)
            
            embed = embedding.fit_transform(dm)[:,0]
            
            dominant_strain = dm.index[np.argsort(embed)[len(embed)//2]]
            
        else:
            raise ValueError('Not a correct type: {}'.format(min_type))

        dominant_strains.append(dominant_strain)
        
        
        assert file_to_seqs is not None

        try:
            dominant_strain = int(dominant_strain)
        except:
            pass
        
        seq = ''.join(file_to_seqs[name].iloc[dominant_strain])
        accession_name = '____'
           
        accession_names.append(accession_name)
        seqs.append(seq)
        
        
    data = pd.DataFrame({
        'name': names,
        'dominant_strains': dominant_strains,
        'sequence': seqs,
        'accession_name': accession_names
    })
    
    data.sort_values(by='name', inplace=True)
    data.reset_index(inplace=True, drop=True)
    
    return data


# ### Merge Predictions

# In[17]:


def compute_ldistances(seqs1, seqs2, max_size=None):
    """
    
    NOTE: if seq1 or seq2 is blank, we return -1.
    """
    
    assert len(seqs1) == len(seqs2)
    
    ldist = []
    for i, seq1 in enumerate(seqs1):
        seq2 = seqs2[i]
        
        if seq1 == '-1' or seq2 == '-1':
            dist = -1
        else:
            dist = Levenshtein.distance(seq1[:max_size], seq2[:max_size])
            
        ldist.append(dist)
        
    return ldist


def merge_prediction_data(
    WHO_rec, 
    qnet_rec, dominant_strains, subtype, 
    f_to_seqs,
    outfile=None,
    max_size=None):
    
    WHO_rec = WHO_rec.reset_index(drop=True)
    qnet_rec = qnet_rec.reset_index(drop=True)
    dominant_strains = dominant_strains.reset_index(drop=True)
    
    data = pd.DataFrame({'year': WHO_rec['year'].values})
    data['WHO_recommendation_name'] = WHO_rec['name']
    data['WHO_recommendation_sequence'] = WHO_rec[subtype]
    
    data['dominant_strain_accession'] = list(dominant_strains['dominant_strains'].values[1:]) + ['-1']
    data['dominant_strain_sequence'] = list(dominant_strains['sequence'].values[1:]) + ['-1']
    
    dom_strain_acc_name = list(dominant_strains['accession_name'].values[1:])
#     dom_strain_acc_name = list(map(parse_influenza_name, dom_strain_acc_name))
    dom_strain_acc_name = dom_strain_acc_name + ['-1']
    data['dominant_strain_accession_name'] = dom_strain_acc_name
    
    data['qdistance_recommendation_accession'] = qnet_rec['dominant_strains']
    data['qdistance_recommendation_sequence'] = qnet_rec['sequence']
    
#     qdist_acc_name = list(map(parse_influenza_name, qnet_rec['accession_name']))
    qdist_acc_name = qnet_rec['accession_name']
    data['qdistance_recommendation_accession_name'] = qdist_acc_name
    
#     import pdb; pdb.set_trace()
    ldistance_WHO = compute_ldistances(
        data['WHO_recommendation_sequence'][:-1],
        data['dominant_strain_sequence'][:-1],
        max_size=max_size)
    
    ldistance_qnet_rec = compute_ldistances(
        data['qdistance_recommendation_sequence'][:-1],
        data['dominant_strain_sequence'][:-1],
        max_size=max_size)
    
    data['ldistance_WHO'] = ldistance_WHO + [-1]
    data['ldistance_Qnet_recommendation'] = ldistance_qnet_rec + [-1]
    
    num_rows = data.shape[0]
    num_samples = []
    for i in range(num_rows):
        year_range = data['year'].iloc[i]
        year = year_range.split('_')[0]
        if year_range in f_to_seqs:
            num_samples.append(f_to_seqs[year_range].shape[0])
        else:
            num_samples.append(-1)
            
    data['qnet_sample_size'] = num_samples
            
    if outfile is not None:
        data.to_csv(outfile, index=None)
    return data


# ## Computation

# In[18]:


NUM_CLUSTERS = 1
NUM_CLUSTERS_QDIST = 3


# In[19]:


if COMPUTE_COMMON_STRAIN:

#     ha_dominant_strains_ldist = compute_dominant_strain(
#         remove_index_and_cols(human_ha_ldistance_dm),
#         NUM_CLUSTERS,
#         None,
#         HUMAN_HA_YEARLY_DIST_MATRIX_LDISTANCE_DIR,
#         use_accession=False, 
#         file_to_seqs=human_ha_seqs, 
#         file_to_seqs_base_dir='{}human'.format(TYPE, TYPE))
    
    ha_dominant_strains_ldist = compute_dominant_strain(
        name_to_dm=remove_index_and_cols(human_ha_ldistance_dm), 
        upper_diag=True,
        n_clusters=NUM_CLUSTERS, 
        file_to_seqs=human_ha_seqs, 
        remove_outliers=False)
    
    ha_dominant_strains_qdist = compute_dominant_strain(
        name_to_dm=remove_index_and_cols(human_ha_qdistance_dm), 
        upper_diag=False,
        n_clusters=1, 
        file_to_seqs=human_ha_seqs, 
        remove_outliers=False)
    


# In[20]:


WHO_recommendations = pd.read_csv(
    '/project2/ishanu/hiv-dip/influenza/data/WHO_recommendations_Northern_Hemisphere.csv')


# In[21]:


if COMPUTE_COMMON_STRAIN:
    human_ha_max_len = 550
    humanHA_recommendations = merge_prediction_data(
        WHO_recommendations, 
        ha_dominant_strains_qdist, 
        ha_dominant_strains_ldist, 
        'HA_seq',
        human_ha_seqs,
        outfile=None,
        max_size=human_ha_max_len)


# # Other
