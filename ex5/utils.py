import numpy as np
from scipy import optimize

def learning_curve(X, y, Xval, yval, lamda):
    """
    Given ...
    X: m,n matrix (without bias term) (input for samples in training set)
    Xval: mval,n matrix (without bias term) (input for samples in validation set)
    y: m,1 matrix (output for samples in training set)
    yval: mval,1 matrix (output for samples in validatoin set)
    lamda: regularization term
    
    Return ...
    error_train: m,1 matrix ( error/cost in training set, as we increase the number of training samples used)
    error_val: m,1 matrix ( error/cost in validation set, as we increase the number of training samples used)
    These errors are used to print the learning curve.
    
    Where ...
    m: number of samples
    n: number of features 
    """
    m,n = X.shape
    initial_theta = np.ones(n+1)
    
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    
    for i in range(0,m):
        theta_optimized = train_linear_reg(initial_theta, X[0:i+1], y[0:i+1], lamda)
        error_train[i] = error_function(theta_optimized, X[0:i+1], y[0:i+1])
        error_val[i] = error_function(theta_optimized, Xval, yval)
    
    return error_train, error_val

def validation_curve(X, y, Xval, yval, lamdas):
    """
    Given ...
    X: m,n matrix (without bias term) (input for samples in training set)
    Xval: mval,n matrix (without bias term) (input for samples in validation set)
    y: m,1 matrix (output for samples in training set)
    yval: mval,1 matrix (output for samples in validatoin set)
    lamdas: a list of different lamda values to try
    
    Return ...
    error_train: lamdas.size,1 matrix ( error/cost in training set, as we use a different lamda value in lamdas list)
    error_val: lamdas.size,1 matrix ( error/cost in validation set, as we use a different lamda value in lamdas list)
    These errors are used to print the learning curve.
    
    Where ...
    m: number of samples
    n: number of features 
    """
    m,n = X.shape
    initial_theta = np.ones(n+1)
    
    error_train = np.zeros(lamdas.size)
    error_val = np.zeros(lamdas.size)
    
    for l in range(0,lamdas.size):
        theta_optimized = train_linear_reg(initial_theta, X, y, lamdas[l])
        error_train[l] = error_function(theta_optimized, X, y)
        error_val[l] = error_function(theta_optimized, Xval, yval)
    
    return error_train, error_val

def map_poly_features(X, highest_degree):
    """
    Given ...
    X: m,n matrix (without bias term)
    highest_degree: the highest polynomial degree to expand the features into.
    
    Return ...
    X_poly: m,degree matrix
    
    Where ...
    m: number of samples
    n: number of features 
    """
    m,n = X.shape
    assert n == 1 # This function only supports feature mapping on 1 input parameter.
    assert highest_degree >= 2
    
    X_poly = np.copy(X)
    for d in range(2,highest_degree+1):
        X_poly = np.hstack(( X_poly, (X_poly[:,0] ** d).reshape(m,1) ))
    return X_poly

def feature_normalization(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu)/sigma
    return X_norm, mu.reshape(-1), sigma.reshape(-1)

def train_linear_reg(initial_theta, X, y, lamda=0.0):
    results = optimize.fmin_cg( cost_function, 
                                fprime=gradient_function, 
                                x0=initial_theta, 
                                args=(X, y, lamda), 
                                maxiter=50, disp=False, full_output=True )
    theta_optimized = results[0]
    min_cost = results[1]
    return theta_optimized

def predictions(theta, X):
    """
    Given ...
    theta: 1d array with n+1 elements (the optimized theta)
    X: m,n matrix (without bias term)
    
    Return ...
    hypothesis: 1d array with m elements
    """
    m,n = X.shape
    X_bias = np.hstack(( np.ones((m,1)), X ))
    p_theta = theta.reshape(( n+1,1 ))
    hypo = _hypothesis(p_theta, X_bias)
    return hypo.reshape(-1)

def error_function(theta, X, y):
    return cost_function(theta, X, y, lamda=0.0)

def cost_function(theta, X, y, lamda=0.0):
    """
    Given ...
    theta: 1d array with n+1 elements
    X: m,n matrix (without bias term)
    y: m,1 matrix
    lamda: scalar
    
    Return ...
    cost: scalar value
    
    Where ...
    m: number of samples
    n: number of features
    """
    m,n = X.shape
    X_bias = np.hstack(( np.ones((m,1)), X ))
    p_theta = theta.reshape(( n+1,1 ))
    
    term = _hypothesis(p_theta, X_bias) - y
    reg_term = lamda / (2 * float(m)) * np.sum( p_theta[1:] ** 2 )
    cost = (1.0 / (2 * m)) * term.transpose().dot(term) + reg_term
    return cost[0,0]

def gradient_function(theta, X, y, lamda=0.0):
    """
    Given ...
    theta: 1d array with n+1 elements
    X: m,n matrix (without bias term)
    y: m,1 matrix
    lamda: scalar
    
    Return ...
    gradients: 1d array with n+1 elements
    
    Where ...
    m: number of samples
    n: number of features
    """
    m,n = X.shape
    X_bias = np.hstack(( np.ones((m,1)), X ))
    p_theta = theta.reshape(( n+1,1 ))
    
    gradient = 1 / float(m) * X_bias.transpose().dot(_hypothesis(p_theta, X_bias) - y)
    gradient[1:] += lamda / float(m) * p_theta[1:]
    
    return gradient.reshape(-1)

def _hypothesis(theta, X):
    """
    Given ...
    theta: n+1,1 matrix
    X: m,n+1 matrix (with bias term)
    
    Return ...
    hypothesis: m,1 matrix
    
    Where ...
    m: number of samples
    n: number of features
    """
    return X.dot(theta)