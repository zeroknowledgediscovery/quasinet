from quasinet.qnet import qdistance_matrix, Qnet
import glob
import numpy as np
import json
import subprocess
import pandas as pd


def extract_diagonal_blocks(M, L):
    blocks = []
    start_row = 0
    start_col = 0

    for l in L:
        block = M[start_row:start_row+l, start_col:start_col+l]
        blocks.append(block)
        
        start_row += l
        start_col += l

    return blocks


def remove_suffix(s):
    if s.endswith('.H'):
        return s[:-2]
    elif s.endswith('.'):
        return s[:-1]
    return s

def get_description_curl(code):
    code=remove_suffix(code)
    url = f"http://icd10api.com/?code={code}&desc=short&r=json"
    #url = f"https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?sf=code,name&terms={code}&maxList=1"
    response = subprocess.run(["curl", "-s", url], capture_output=True, text=True)
    data = json.loads(response.stdout)
    #print(data)
    #print(data.keys())
    if data['Response']:
        return data['Description']
    else:
        return code


def replace_with_d(S, j, d):
    d_array = np.array(d)[:, np.newaxis]
    output = np.tile(S, (len(d), 1))
    output[:, j] = d_array[:, 0]
    return output









class zQnet(Qnet):
    """Extended Qnet architecture (`zQnet`).

    An extension of the Qnet class with added functionality and attributes. This class
    introduces risk computation based on a series of metrics and provides a way to 
    set and retrieve a model description.

    Inherits from
    -------------
    Qnet : Base class

    New Attributes
    --------------
    nullsequences : array-like
        Collection of sequences considered null or baseline for risk calculations.

    target : str, optional
        Target variable or description. Not currently utilized in methods.

    description : str
        Descriptive notes or commentary about the model.

    auc_estimate : array-like
        AUCs obtained during optimization of null sequences.

    training_index : array-like, optional
        Indices used during the training phase. Not currently utilized in methods.

    Parameters
    ----------
    *args :
        Variable length argument list inherited from Qnet.

    **kwargs :
        Arbitrary keyword arguments inherited from Qnet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the parent's constructor

        self.nullsequences = None
        self.target = None
        self.description = None
        self.training_index = None
        self.auc_estimate = None
        
    def risk_median(self, X):
        """
        Compute the median risk value for input X based on its distance 
        from null sequences.

        Parameters
        ----------
        X : 2d array-like
            Input data whose risk is to be computed.

        Returns
        -------
        float
            Median risk value for the input X.
        """
        return np.median(qdistance_matrix(X, self.nullsequences, self, self))
    
    def risk(self, X):
        """
        Compute the mean risk value for input X based on its distance 
        from null sequences.

        Parameters
        ----------
        X : 2d array-like
            Input data whose risk is to be computed.

        Returns
        -------
        float
            Mean risk value for the input X.
        """
        return np.mean(qdistance_matrix(X, self.nullsequences, self, self))
    
    def risk_max(self, X):
        """
        Compute the maximum risk value for input X based on its distance 
        from null sequences.

        Parameters
        ----------
        X : 2d array-like
            Input data whose risk is to be computed.

        Returns
        -------
        float
            Maximum risk value for the input X.
        """
        return np.max(qdistance_matrix(X, self.nullsequences, self, self))

    def set_description(self, markdown_file):
        """
        Set the description attribute for the model using content from a markdown file.

        Parameters
        ----------
        markdown_file : str
            Path to the markdown file containing the model's description.

        Returns
        -------
        str
            Content of the markdown file.
        """
        with open(markdown_file, 'r') as f:
            content = f.read()
            self.description = content
        return content


    def zshap(self,seq=None,m=35):
        """
        A superfast approximation of SHAP for zQnet

        Parameters
        ----------
        seq : numpy array of str
            The sequence around which we are evaluating perturbations. 
            By default it is the array oif empty strings, which represents average behavior
        m : int
            Length of shap return dataframe
        Returns
        -------
        pandas.DataFrame
            dataframe with shapo values and index mapped to short description of icd10 codes
        """
        
        if seq is None:
            seq=np.array(['']*len(self.feature_names))
        pdist=self.predict_distributions(seq)
        L=[len(pdist[i].keys()) for i in range(len(pdist))]
        A=[]
        for index_ in range(len(seq)):
            A.extend(replace_with_d(seq, index_, list(pdist[index_].keys())))
        D=qdistance_matrix(np.array(A),np.array(A),self,self)
        Shm=[np.std(m) for m in extract_diagonal_blocks(D, L)]
        sf=pd.DataFrame(Shm,self.feature_names,columns=['shap']).sort_values('shap').tail(m)
        sf['code']=sf.index.values
        sf.index=[get_description_curl(code) for code in sf.index.values]
        return sf

    def personal_zshap(self,s,eps=1e-7):
        """
        A superfast approximation of SHAP for zQnet for individual samples 

        Parameters
        ----------
        s : numpy array of str
            The sequence around which we are evaluating perturbations. 
        eps : float
            shap value cutoff
        Returns
        -------
        pandas.DataFrame
            dataframe with shapo values and index mapped to short description of icd10 codes
        """
        
        rs=self.risk_max(np.array([s]))
        sD={self.feature_names[i]:rs-self.risk_max(np.array([np.where(np.arange(len(s)) == i,
                                                                      'H', s)])) for i in np.where(s!='H')[0]}
        sf=pd.DataFrame(sD,index=['shap']).transpose()
        sf=sf[sf.abs()>eps].dropna()
        sf.index=[x+y for (x,y) in zip(sf.index.values,
                                       s[np.array([np.where(self.feature_names==i)[0][0] for i in sf.index.values])])]
        sf['code']=sf.index.values
        sf.index=[get_description_curl(code) for code in sf.index.values]
        return sf



