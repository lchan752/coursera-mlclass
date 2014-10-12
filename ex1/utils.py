import numpy as np

def hypothesis(X, theta):
    """
    assuming there are m samples and n features
    X is m,n+1 matrix
    theta is n+1,1 matrix
    returns hypothesis, which is a m,1 matrix
    """
    return X.dot(theta)

def computeCostLoop(X, y, theta):
    m = len(y)
    cumulative_sum = 0
    for i in range(0,m):
        cumulative_sum += (hypothesis(X[i], theta) - y[i]) ** 2
    cumulative_sum = (1.0 / (2 * m)) * cumulative_sum
    return cumulative_sum

def computeCost(X, y, theta):
    m = len(y)
    #return (1.0 / (2 * m))*np.sum( (X.dot(theta) - y) ** 2 )
    term = hypothesis(X, theta) - y
    return (1.0 / (2 * m)) * term.transpose().dot(term)[0,0]

def gradientDescent(X, y, theta, alpha, iterations):
    cnt = 1
    m = len(y)
    iter_cnts = []
    iter_costs = []
    while cnt<=iterations:
        iter_cnts.append(cnt)
        iter_costs.append(computeCost(X, y, theta))
        term = hypothesis(X, theta) - y
        theta = theta - alpha * (1/float(m)) * X.transpose().dot(term)
        cnt += 1
    
    # Can plot iter_costs against iter_cnts. Cost should decrease on every iteration, until reaches a plateau
    # If cost goes up, it means alpha is too big, need to reduce alpha
    #import matplotlib.pyplot as plt
    #plt.plot(iter_cnts,iter_costs)
    
    # TODO: Try implement gradientDescent with scipy.optimize, and use different optimization functions.
    return theta

def feature_normalization_loop(X):
    m, n = np.shape(X)
    mu = np.zeros((n,1))
    sigma = np.zeros((n,1))
    X_norm = np.zeros(np.shape(X), dtype=X.dtype)
    
    for col in range(0,n):
        mu[col] = np.mean(X[:,col])
        sigma[col] = np.std(X[:,col], ddof=1)
        X_norm[:,col] = [ (x-mu[col])/sigma[col] for x in X[:,col] ]
    
    return X_norm, mu, sigma

def feature_normalization(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu)/sigma
    return X_norm, mu.reshape((2,1)), sigma.reshape((2,1))

def normal_equation(X, y):
    """
    Assume there are m samples and n features.
    X is m,n+1 matrix
    y is m,1 matrix
    return theta, which is a n+1,1 matrix
    """
    return np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
