import math
import numpy as np
from scipy import special,optimize,misc
import matplotlib.pyplot as plt

def print_number(X):
    img = misc.toimage(X[1:].reshape(20,20))
    figure  = plt.figure()
    axes    = figure.add_subplot(111)
    axes.imshow( img )
    plt.show()

def sigmoid(z):
    """
    Returns sigmoid vector
    z: m,1 matrix
    returns: m,1 matrix
    """
    return special.expit(z)

def hypothesis(X, theta):
    """
    Returns hypothesis vector
    Assuming there are m samples, n features...
    X: m,(n+1) matrix
    theta: (n+1),1 matrix
    returns: m,1 matrix
    """
    return sigmoid(X.dot(theta))

def cost_function(theta, X, y, lamda):
    m = X.shape[0]
    p_theta = theta.reshape((theta.size),1)
    term1 = y.transpose().dot(np.log(hypothesis(X, p_theta)))
    term2 = (1-y).transpose().dot(np.log( 1 - hypothesis(X, p_theta) ))
    term3 = p_theta[1:].transpose().dot(p_theta[1:])
    #term3 = p_theta.transpose().dot(p_theta)
    cost = - (1/float(m)) * ( term1 + term2 ) + lamda/(2*float(m)) * term3
    return cost[0,0]

def cost_gradient(theta, X, y, lamda):
    m = X.shape[0]
    p_theta = theta.reshape((theta.size),1)
    gradient = (1/float(m)) * X.transpose().dot( (hypothesis(X, p_theta) - y)) 
    gradient[1:] = gradient[1:] + ( (p_theta[1:] * lamda ) / m )
    return gradient.reshape(-1)

def gradient_descent(theta, X, y, lamda):
    results = optimize.fmin_cg( cost_function, fprime=cost_gradient, x0=theta, args=(X, y, lamda), maxiter=50, disp=False, full_output=True )
    return results[0], results[1]

def classifyall(initial_theta, X, y, lamda):
    m = X.shape[0]
    n = X.shape[1]-1
    K = y.shape[1]
    theta = None
    
    for k in range(0,K):
        y_k = y[:,k].reshape((y[:,k].size,1))
        initial_theta_k = initial_theta[:,k]
        theta_k, min_cost = gradient_descent(initial_theta_k, X, y_k, lamda)
        print "Min Cost for Class {}: {}".format(k,min_cost)
        theta = theta_k.reshape((n+1,1)) if theta is None else np.hstack((theta,theta_k.reshape((n+1,1))))
    
    return theta

def accuracy(predicted_y, expected_y):
    m = predicted_y.size
    difference = predicted_y[predicted_y != expected_y].size
    return (1-difference/float(m))*100

def feedforward(X,theta1,theta2):
    """
    Return m predictions from this neural network.
    m = number of samples = X.shape[0]
    Each prediction is a number between 0,K (for K classes)
    """
    m,_ = X.shape
    retlist = []
    for s in range(0,m):
        a_1 = X[s,:] # (n+1,1)
        a_2 = sigmoid(theta1.dot(a_1)) # (25,1)
        a_2 = np.insert(a_2,0,1,axis=0) # (26,1)
        a_3 = sigmoid(theta2.dot(a_2)) # (10,1)
        retlist.append(np.argmax(a_3))
    return np.array(retlist)
        