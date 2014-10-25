import numpy as np

def gaussian_kernel(x1,x2,sigma):
    """
    return Gaussian distance between 2 sample
    
    Given ...
    x1: 1d array, x1.size == n number of features
    x2: 1d array, x2.size == n number of features
    sigma: bandwidth parameter
    
    Returns ...
    Gaussian distance (scalar value)
    """
    assert x1.size == x2.size
    diff = x1 - x2
    return np.exp( -1/float(2*sigma**2) * np.sum(diff**2))