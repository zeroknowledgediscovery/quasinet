from .qnet import load_qnet
from .qsampling import qsample
from .qnet import qdistance
from .metrics import theta
from .qnet import Qnet
from .utils import getNull

from tqdm import tqdm
import numpy as np
import copy
from concurrent.futures import ProcessPoolExecutor
 

def dist_scalr_mult(D1,a):
    """
    Multiply each value in the dictionary with scalar 'a' and renormalize 
    to get a valid probability distribution.

    Parameters
    ----------
    D1 : dict
        Dictionary where each key-value pair represents an item and its probability.
    a : float
        Scalar to multiply with each value of D1.

    Returns
    -------
    dict
        New dictionary with each value scaled and renormalized.
    """
    S=0.0
    D={}
    for i in D1.keys():
        S+=D1[i]**a
    for i in D1.keys():
        D[i]=(D1[i]**a)/S
    return D

def dist_sum(D1,D2):
    """
    Add each corresponding value in D1 and D2, then renormalize to get 
    a valid probability distribution.
    
    Parameters
    ----------
    D1, D2 : dict
        Two dictionaries where each key-value pair represents an item 
        and its probability.

    Returns
    -------
    dict
        New dictionary with each value being the sum of the corresponding 
        values in D1 and D2, renormalized.
    """
    S=0.0
    D={}
    for i in D1.keys():
        S+=D1[i]*D2[i]
    for i in D1.keys():
        D[i]=(D1[i]*D2[i])/S
    return D
  
def distance_function(p, q,NULL=None,strtype='U5'):
    """
    Computes the distance between two Quasinets.

    Parameters:
    p, q (Quasinet): The Quasinets to compute the distance between.
 
    Returns:
    float: The distance between p and q.
    """
    if NULL is None:
        NULL=getNull(p,strtype=strtype)
    return qdistance(NULL,NULL,p,q)


def scalarmod_predict_distribution(self, column_to_item, column, **kwargs):
    """
    Modify the predict_distribution function of the Quasinet object
    to scale the output probabilities for a specified feature.

    Parameters
    ----------
    self : Quasinet object
        The Quasinet instance.
    column_to_item : dict
        A dictionary mapping from column names to specific items.
    column : str
        The name of the column (feature) to scale.
    **kwargs : dict
        Additional arguments passed to the predict_distribution function.

    Returns
    -------
    dict
        A dictionary of probabilities for each item in the specified column.
    """
    distrib = Qnet.predict_distribution(self, column_to_item, column, **kwargs)
    if column == self.direction:
        distrib = dist_scalr_mult(distrib,self.delta)

    return distrib

def sum_predict_distribution(self, column_to_item, column, **kwargs):
    distrib = Qnet.predict_distribution(self, column_to_item, column, **kwargs)
    if column == self.direction:
        return dist_sum(distrib,dist_scalr_mult(distrib,self.delta))
    return distrib


def delta_pi(qnet_instance,index,delta):
    """
    This function modifies the distribution of the given Quasinet instance by scaling it 
    with a scalar value in the direction of the given index.

    Parameters:
    qnet_instance (Quasinet): The Quasinet instance to modify.
    index (int): The index of the feature direction to scale.
    delta (float): The scalar to scale the distribution with.

    Returns:
    Quasinet: The Quasinet instance with modified distribution.
    """
    p=copy.copy(qnet_instance)
    p.predict_distribution = scalarmod_predict_distribution.__get__(p)
    p.direction = index
    p.delta = delta
    return p

def perturb_quasinet(qnet_instance,index,delta):
    """
    Perturbs a Quasinet in the direction of the i-th feature.

    Parameters:
    p (list[dict(str,flaot]): quasinet.predict_distributions()
    i (int): The index of the feature direction to perturb in.
    delta (float): The magnitude of the perturbation.

    Returns:
    Quasinet: The perturbed Quasinet.
    """
    
    p=copy.copy(qnet_instance)
    p.predict_distribution = sum_predict_distribution.__get__(p)
    p.direction = index
    p.delta = delta
    return p

def perturb_quasinet_distrib(p_distrib_,index,delta):
    """
    Perturbs a Quasinet in the direction of the i-th feature, using only the distributions at 
    each estimator, which are produced by the predict_distributions function

    Parameters:
    p (list[dict(str,flaot]): quasinet.predict_distributions()
    i (int): The index of the feature direction to perturb in.
    delta (float): The magnitude of the perturbation.

    Returns:
    Quasinet: The perturbed Quasinet.
    """
    p_distrib=copy.copy(p_distrib_)
    p_distrib[index] = dist_sum(p_distrib_[index],dist_scalr_mult(p_distrib_[index],delta))
    return np.array(p_distrib)


def distance_function_distrib(p, q,i):
    """
    Compute distance between two quasinets assumeing that p, q only differ at  estimator coordinates listed in i
    Parameters:
    p (list[dict(str,flaot]): quasinet.predict_distributions()
    q (list[dict(str,flaot]): quasinet.predict_distributions()
    i (1d numpy arra): list of indices on which p and q differ


    """
    return theta(p[i],q[i])*(len(i)/len(p))
   		

def mt_worker(args):
    p_distrib, delta, i, n = args
    row = np.zeros(n)
    Pi=perturb_quasinet_distrib(p_distrib, i, delta)
    dpi = distance_function_distrib(Pi, p_distrib,[i])
    for j in range(i):
        Pj=perturb_quasinet_distrib(p_distrib, j, delta)
        
        row[j] = 0.5 * (distance_function_distrib(Pi, Pj,[i,j])**2 -
                        dpi**2 -
                        distance_function_distrib(Pj, p_distrib,[j])**2) / delta**2
    return row

def compute_metric_tensor(p_distrib, delta, progress=False):
    """
    Computes the metric tensor at a given point in the space of Quasinets.

    The metric tensor G_ij is defined as:

    .. math::
        G_{ij} = \\frac{1}{2} \\left( D(p + \\delta p_i + \\delta p_j, p) - D(p + \\delta p_i, p) - D(p + \\delta p_j, p) + D(p, p) \\right)

    where D is the distance function, p_i is the i-th unit Quasinet, and delta is a small perturbation.

    Parameters:
    p_distrib (list[dict(str,float]): quasinet.predict_distributions() for the quasinet 'p' at which metric tensor is calculated
    delta (float): A small number representing a change in each coordinate direction.
    progress (bool): show progress bar

    Returns:
    ndarray: The metric tensor at point p (the quasinet for which p_distrib is calculatd).
    """
    n = len(p_distrib)
    with ProcessPoolExecutor() as executor:
        args = [(p_distrib, delta, i, n) for i in range(n)]
        if progress:
            G = list(tqdm(executor.map(mt_worker, args), total=n))
        else:
            G = list(executor.map(mt_worker, args))
    return np.array(G)+np.array(G).T


def compute_metric_tensor_derivative(p, delta):
    """
    Computes the derivative of the metric tensor at a given point in the space of Quasinets.

    The derivative of the metric tensor G_ij with respect to the k-th coordinate is computed as:

    .. math::
        \\frac{\\partial G_{ij}}{\\partial p_k} = \\frac{G_{ij}(p + \\delta p_k) - G_{ij}(p)}{\\delta}

    Parameters:
    p (list[dict(str,float]):  quasinet.predict_distributions() for the quasinet 'p' at which compute the metric tensor derivative.
    delta (float): A small number representing a change in each coordinate direction.

    Returns:
    ndarray: The derivative of the metric tensor at point p.
    """
    n = len(p)
    G = compute_metric_tensor(p, delta)
    G_prime = np.zeros((n, n, n))
    for k in range(n):
        G_shifted = compute_metric_tensor(perturb_quasinet_distrib(p, k, delta), delta)
        G_prime[:, :, k] = (G_shifted - G) / delta
    return G_prime


def compute_ricci_curvature(p, delta):
    """
    Computes the Ricci curvature at a given point in the space of Quasinets.

    The Ricci curvature R_ij is computed as:

    .. math::
        R_{ij} = G^{kl} \\left( \\frac{\\partial^2 G_{ij}}{\\partial p_k \\partial p_l} - \\frac{1}{2} \\frac{\\partial^2 G_{kl}}{\\partial p_i \\partial p_j} \\right)

    where G^{kl} is the inverse of the metric tensor G_{kl}, and the partial derivatives are computed by taking the limit as delta goes to zero.

    Parameters:
    p (list[dict(str,float]):  quasinet.predict_distributions() for the quasinet 'p' 
    delta (float): A small number representing a change in each coordinate direction.

    Returns:
    ndarray: The Ricci curvature at point p.
    """
    n = len(p)
    G = compute_metric_tensor(p, delta)
    G_inv = np.linalg.inv(G)
    G_prime = compute_metric_tensor_derivative(p, delta)
    G_double_prime = np.zeros((n, n, n, n))
    for l in tqdm(range(n)):
        G_prime_shifted = compute_metric_tensor_derivative(perturb_quasinet_distrib(p, l, delta), delta)
        G_double_prime[:, :, :, l] = (G_prime_shifted - G_prime) / delta
    ricci_curvature = np.zeros((n, n))
    for i in tqdm(range(n)):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    ricci_curvature[i, j] += G_inv[k, l] * (G_double_prime[i, k, j, l] - 0.5 * G_double_prime[k, l, i, j])
    return ricci_curvature

def compute_curvature(p, delta):
    """
    Computes the curvature (scalar curvature) at a given point in the space of Quasinets.

    The curvature R is computed as:

    .. math::
        R = G^{ij} R_{ij}

    where G^{ij} is the inverse of the metric tensor G_{ij}, and R_{ij} is the Ricci curvature.

    Parameters:
    p (list[dict(str,float]):  quasinet.predict_distributions() for the quasinet 'p' 
    delta (float): A small number representing a change in each coordinate direction.

    Returns:
    float: The curvature at point p.
    """
    n = len(p)
    G = compute_metric_tensor(p, delta)
    G_inv = np.linalg.inv(G)
    ricci_curvature = compute_ricci_curvature(p, delta)
    curvature = 0
    for i in tqdm(range(n)):
        for j in range(n):
            curvature += G_inv[i, j] * ricci_curvature[i, j]
    return curvature
