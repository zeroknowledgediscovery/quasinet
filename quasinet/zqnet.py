from quasinet.qnet import qdistance_matrix, Qnet
import numpy as np

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

    training_index : array-like, optional
        Indices used during the training phase. Not currently utilized in methods.

    Methods
    -------
    risk_median(X)
        Compute the median risk value for the given input X.

    risk(X)
        Compute the mean risk value for the given input X.

    risk_max(X)
        Compute the maximum risk value for the given input X.

    set_description(markdown_file)
        Set the model's description attribute using content from a markdown file.

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
