import numpy as np
from .qsampling import qsample
from .qnet import qdistance
from .qnet import qdistance_matrix
from .utils import getNull
from catboost import CatBoostRegressor
import shap
import pandas as pd


global NULL
global model

def qnet_model_func(X):
    """
    Function to compute the distance matrix for a given set of sequences.

    Parameters
    ----------
    X : numpy.ndarray
        Array of sequences.

    Returns
    -------
    numpy.ndarray
        The computed distance matrix.
    """
    global NULL
    global model
    return np.atleast_1d(qdistance_matrix(X, np.array([NULL]), model, model).squeeze())

def getShap(model_, num_backgrounds=1, samples=None, num_samples=5, strtype='U5', fast_estimate=False):
    """
    Function to compute SHAP values for feature importance analysis.

    Parameters
    ----------
    model : Qnet object
        The Qnet model.

    num_backgrounds : int
        Number of background samples to generate. Default is 1.

    num_samples : int
        Number of samples for the SHAP analysis. Default is 5.

    strtype : str
        String type to be used for the generated numpy array. Default is 'U5'.

    samples : numpy array
        samples to run shap analysis on. Default is None. If None, generate via qsampling

    fast_estimate : bool
        If True, use tree explainer with a CatBoostRegressor model 
        for faster estimation. Default is False.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the SHAP values for each feature.
    numpy.array 
        numpy array of ordered indices of decsening shapvalues of model feature_names
    """
    global NULL
    global model

    model=model_
    
    NULL=getNull(model,strtype=strtype)
    background_samples = np.array([qsample(NULL,
                                           model, steps=5000) for _ in range(num_backgrounds)])

    # Generate samples for the SHAP analysis
    if samples is None:
        samples = np.array([qsample(NULL, model,
                            steps=len(model.feature_names)) for _ in range(num_samples)])

    if fast_estimate:
        # Train a CatBoostRegressor model
        cat_model = CatBoostRegressor(verbose=False)
        cat_model.fit(samples, qnet_model_func(samples))

        explainer = shap.TreeExplainer(cat_model)
    else:
        explainer = shap.KernelExplainer(qnet_model_func,
                                         background_samples)

    shap_values = explainer.shap_values(samples)

    sf=pd.DataFrame(shap_values.mean(axis=0),columns=['shapval'])
    sf.index=model.feature_names
    sf=sf.sort_values('shapval')
    sf['shapvalabs']=sf.shapval.abs()
    sf=pd.DataFrame(sf.sort_values('shapvalabs',ascending=False).shapval)


    xf=pd.DataFrame(np.array(model.feature_names),columns=['feature_names'])
    xf=xf.reset_index().set_index('feature_names').T
    xf=xf[sf.index.values].T
    xf.columns=['id']

    
    return sf,xf.id.values
