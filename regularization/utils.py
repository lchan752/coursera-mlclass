import math
import numpy as np
from scipy import special
from scipy import optimize

# TODO: should make function parameters support 1d arrays, if it is a x,1 matrix. Or else I will have to mess with reshape all the time.

def sigmoid(z):
    """
    Returns sigmoid vector
    z: m,1 matrix
    returns: m,1 matrix
    """
    return special.expit(z)
    #return 1 / ( 1 + np.array(map( lambda t:math.exp(t), -z )) )

def hypothesis(X, theta):
    """
    Returns hypothesis vector
    Assuming there are m samples, n features...
    X: m,(n+1) matrix
    theta: (n+1),1 matrix
    returns: m,1 matrix
    """
    return sigmoid(X.dot(theta))

def map_feature(X1, X2, d):
    """
    maps features to d dimensions
    X1: m,1 matrix
    X2: m,1 matrix
    returns: m,np.sum(range(2,d+2)) matrix
    """
    m = X1.shape[0]
    out = None
    for i in range(1,d+1):
        for j in range(0,i+1):
            term1 = X1 ** (i-j)
            term2 = X2 ** j
            term = (term1 * term2).reshape((m,1))
            out = np.hstack(( out,term )) if out is not None else term
    return out

def regularized_cost_function_loop(X, y, theta, lamda):
    m,n = X.shape
    term1 = 0
    term2 = 0
    for i in range(0,m):
        term1 += - y[i] * math.log(hypothesis(X[i,:], theta)) - ( 1-y[i] ) * math.log(( 1 - hypothesis(X[i,:], theta) ))
    for j in range(1,n):
        term2 += theta[j] ** 2
    return 1/float(m) * term1 + lamda/(2*m) * term2

def regularized_cost_function(X, y, theta, lamda):
    m = X.shape[0]
    term1 = y.transpose().dot(np.log(hypothesis(X, theta)))
    term2 = (1-y).transpose().dot(np.log( 1 - hypothesis(X, theta) ))
    term3 = theta[1:].transpose().dot(theta[1:])
    return - (1/float(m)) * ( term1 + term2 ) + lamda/(2*float(m)) * term3

def regularized_cost_gradient(X, y, theta, lamda):
    m = X.shape[0]
    gradient = (1/float(m)) * X.transpose().dot( (hypothesis(X, theta) - y)) 
    gradient[1:] = gradient[1:] + ( (theta[1:] * lamda ) / m )
    return gradient

def regularized_gradient_descent(X, y, theta, lamda):
    def inner_cost_function(theta, X, y, lamda):
        # just wanted to rearrange the function parameters to fit scipy.optimize's format
        return regularized_cost_function(X,y,theta,lamda)
    results = optimize.minimize(inner_cost_function, theta, args=(X, y, lamda),  method='BFGS', options={"maxiter":500, "disp":True} )
    return results.x, results.fun

def cost_function_loop(X, y, theta):
    """
    Returns cost vector
    For every choice of theta, m samples and n features of X,y, there is one cost.
    So if we have 100 iterations, with 100 choices of thetas, we have 100 costs.
    Here we are calculating the cost for m samples and n features of X,y, and once choice of theta,
    so it only returns 1 cost (so output is 1,1 in shape)
    
    Assuming there are m samples, n features...
    X: m,(n+1) matrix
    y: m,1 matrix
    theta: (n+1),1 matrix
    returns: 1,1 matrix
    """
    m,n = X.shape
    term = 0
    for i in range(0,m):
        term += - y[i] * math.log(hypothesis(X[i,:], theta)) - ( 1-y[i] ) * math.log(( 1 - hypothesis(X[i,:], theta) ))
    return 1/float(m) * term

def cost_function(X, y, theta):
    """
    vectorized version of cost_function_loop
    """
    m = X.shape[0]
    term1 = y.transpose().dot(np.log(hypothesis(X, theta)))
    term2 = (1-y).transpose().dot(np.log( 1 - hypothesis(X, theta) ))
    return - (1/float(m)) * ( term1 + term2 )

def cost_gradient(X, y, theta):
    """
    Returns cost gradient, which is a (n+1),1 matrix
    We take differential of the cost function on each parameter in theta.
    There are n+1 parameters in theta, thus we have n+1 cost gradients.
    X: m,(n+1) matrix
    y: m,1 matrix
    theta: (n+1),1 matrix
    returns: (n+1),1 matrix
    """
    m,n = X.shape
    return (1/float(m)) * X.transpose().dot( (hypothesis(X, theta) - y) )

def gradient_descent(X, y, theta):
    """
    Performs gradient descent with scipy.optimize
    X: m,(n+1) matrix
    y: m,1 matrix
    theta: (n+1),1 matrix
    returns: a tuple, theta and min_cost. theta is (n+1),1 matrix
    """
    
    def inner_cost_function(theta, X, y):
        # just wanted to rearrange the function parameters to fit scipy.optimize's format
        return cost_function(X,y,theta)
    
    results = optimize.fmin(inner_cost_function, x0=theta, args=(X,y), maxiter=500, full_output=True, disp=False )
    return results[0], results[1]

def classify(data, X, theta):
    """
    Classify the data to 'admitted' or 'rejected' class, using the supplied theta
    data: 1,(n+1) matrix, the data to be classified. (e.g. [ 1, testscore1, testscore2 ])
    theta: (n+1),1 matrix, the optimized theta
    X: m,(n+1) matrix, the training data
    returns 1 or 0 (i.e. admitted or rejected)
    """
    m,n = X.shape
    p1 = hypothesis(data, theta)
    return 1 if p1 >= 0.5 else 0