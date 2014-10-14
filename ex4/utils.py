import numpy as np
from scipy import special,optimize,misc

def sigmoid(z):
    return special.expit(z)

def sigmoid_gradient(z):
    return sigmoid(z) * ( np.ones(z.shape) - sigmoid(z) )

def generate_theta(shape, epsilon):
    return np.random.random(shape) * 2 * epsilon - epsilon

def hypothesis_loop(theta1, theta2, X):
    """
    Return hypothesis for a 3-layer neural network
    X: m,n+1, assume x_0 bias term is already added, x_0 = 1
    theta1: s_2,s_1+1 => s_2,n+1
    theta2: s_3,s_2+1 => K,s_2+1, where K is number of output classes
    Return hypothesis: m,K
    """
    p_X = X.reshape((1,X.size)) if X.ndim == 1 else X
    m = p_X.shape[0] # number of samples
    n = p_X.shape[1]-1 # number of features
    s_2 = theta1.shape[0] # number of units in layer 2
    s_1 = theta1.shape[1]-1 # number of units in layer 1
    assert s_1 == n # validate number of units in layer 1 == number of input features
    K = s_3 = theta2.shape[0] # number of output classes K == number of units in layer 3
    assert s_2 == theta2.shape[1]-1 # more input validation
    L = 3 # number of layers
    
    hypo = None
    for i in range(0,m):
        z_2 = theta1.dot(p_X[i,:])
        a_2 = sigmoid(z_2)
        a_2 = np.insert(a_2,0,1) # append bias term
        z_3 = theta2.dot(a_2)
        a_3 = sigmoid(z_3)
        hypo = a_3 if hypo is None else np.vstack((hypo,a_3))
    return hypo

def cost_function_loop(theta1, theta2, X, y):
    m = X.shape[0] # number of samples
    n = X.shape[1]-1 # number of features
    s_2 = theta1.shape[0] # number of units in layer 2
    s_1 = theta1.shape[1]-1 # number of units in layer 1
    assert s_1 == n # validate number of units in layer 1 == number of input features
    K = s_3 = theta2.shape[0] # number of output classes K == number of units in layer 3
    assert s_2 == theta2.shape[1]-1 # more input validation
    L = 3 # number of layers
    
    cost = 0
    for i in range(0,m):
        hypo = hypothesis_loop(theta1, theta2, X[i,:]).reshape((1,K))
        term1 = y[i,:].reshape((1,K)).dot(np.log(hypo).transpose())
        term2 = (np.ones((1,K))-y[i,:]).reshape((1,K)).dot(np.log(np.ones((1,K)) - hypo).transpose())
        cost += -(1/float(m)) * (term1 + term2)
    return cost[0,0]

def regularized_cost_function_loop(theta1, theta2, X, y, lamda):
    m = X.shape[0] # number of samples
    n = X.shape[1]-1 # number of features
    s_2 = theta1.shape[0] # number of units in layer 2
    s_1 = theta1.shape[1]-1 # number of units in layer 1
    assert s_1 == n # validate number of units in layer 1 == number of input features
    K = s_3 = theta2.shape[0] # number of output classes K == number of units in layer 3
    assert s_2 == theta2.shape[1]-1 # more input validation
    L = 3 # number of layers
    
    cost = cost_function_loop(theta1, theta2, X, y)
    reg_term = np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2)
    cost_reg = cost + reg_term * lamda/( 2*float(m) )
    return cost_reg

def rolltheta(theta, input_layer_size, hidden_layer_size, num_labels):
    theta1_shape = (hidden_layer_size, input_layer_size+1)
    theta2_shape = (num_labels, hidden_layer_size+1)
    theta1_size = theta1_shape[0] * theta1_shape[1]
    theta2_size = theta2_shape[0] * theta2_shape[1]
    theta1 = theta[:theta1_size].reshape(theta1_shape)
    theta2 = theta[theta1_size:theta1_size+theta2_size].reshape(theta2_shape)
    return (theta1, theta2)

def unrolltheta(theta1, theta2):
    return np.hstack((theta1.flatten(), theta2.flatten()))

def nnCostFunction(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamda=0):
    """
    Return the cost. (scalar value)
    
    Given ...
    theta: unrolled 1d array
    nn layer sizes
    num_labels: number of output classes (equals K)
    X: m,n (DON'T include bias term in input X)
    y: m,K (vectorized representation for each y)
    lamda
    
    Returns ...
    cost: single scalar value.
    
    Where ...
    m = number of samples
    n = number of input features
    K = number of output classes
    """
    m, n = X.shape
    X_bias = np.hstack((np.ones((m,1)), X))
    theta1, theta2 = rolltheta(theta, input_layer_size, hidden_layer_size, num_labels)
    a1,a2,a3,z2,z3 = _feedforward(theta1, theta2, X_bias)
    
    term1 = -y * np.log(a3) # m,K
    term2 = (1-y) * np.log(1-a3) # m,K
    reg_term = lamda/(2*float(m)) * (np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2)) # scalar
    cost = np.sum((term1 - term2)/float(m)) + reg_term # scalar
    
    return cost

def nnGradientFunction(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamda=0):
    """
    return gradients of the cost function as an unrolled 1d array
    
    Given ...
    theta: unrolled 1d array
    nn layer sizes
    num_labels: number of output classes (equals K)
    X: m,n (DON'T include bias term in input X)
    y: m,K (vectorized representation for each y)
    lamda
    
    Returns ...
    gradient: 1d array (gradient.size == theta1.size + theta2.size)
    
    Where ...
    m = number of samples
    n = number of input features
    K = number of output classes
    """
    m, n = X.shape
    K = y.shape[1]
    X_bias = np.hstack((np.ones((m,1)), X))
    theta1, theta2 = rolltheta(theta, input_layer_size, hidden_layer_size, num_labels)
    a1,a2,a3,z2,z3 = _feedforward(theta1, theta2, X_bias)
    
    delta_accu1 = np.zeros(theta1.shape)
    delta_accu2 = np.zeros(theta2.shape)
    
    for t in range(0,m):
        delta3 = (a3[t,:] - y[t,:]).reshape((K,1))
        a2_bias = np.insert(a2[t,:],0,1).reshape((a2[t,:].size+1,1))
        delta2 = theta2.transpose().dot(delta3) * a2_bias * (1-a2_bias)
        a1_bias = a1[t,:].reshape((n+1,1))
        delta_accu1 = delta_accu1 + delta2[1:].dot(a1_bias.transpose())
        delta_accu2 = delta_accu2 + delta3.dot(a2_bias.transpose())
    
    gradient1 = 1/float(m) * delta_accu1
    gradient2 = 1/float(m) * delta_accu2
    
    gradient1[:,1:] += lamda / float(m) * theta1[:,1:]
    gradient2[:,1:] += lamda / float(m) * theta2[:,1:]
    
    return unrolltheta(gradient1, gradient2)

def nnTrain(initial_theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamda=0):
    """
    Trains a neural network, returns theta1, theta2
    
    Given:
    initial_theta: unrolled randomized theta that breaks symmetry
    nn layer sizes
    num_labels: number of output classes (equals K)
    X: m,n (DON'T include bias term in input X)
    y: m,K (vectorized representation for each y)
    lamda
    
    Returns:
    theta1, theta2
    """
    
    results = optimize.fmin_cg( nnCostFunction, 
                                fprime=nnGradientFunction, 
                                x0=initial_theta, 
                                args=(input_layer_size, hidden_layer_size, num_labels, X, y, lamda), 
                                maxiter=50, disp=False, full_output=True )
    theta_optimized = results[0]
    min_cost = results[1]
    rolled_theta_optimized = rolltheta(theta_optimized, input_layer_size, hidden_layer_size, num_labels)
    return rolled_theta_optimized[0], rolled_theta_optimized[1]

def nnPredict(theta1, theta2, X):
    m, n = X.shape
    X_bias = np.hstack((np.ones((m,1)), X))
    a1,a2,a3,z2,z3 = _feedforward(theta1, theta2, X_bias)
    return np.argmax(a3, axis=1)

def accuracy(predicted_y, expected_y):
    m = predicted_y.size
    difference = predicted_y[predicted_y != expected_y].size
    return (1-difference/float(m))*100

def _feedforward(theta1, theta2, X):
    """
    returns a1,a2,a3,z2,z3 of neural network
    
    Given ...
    theta1: s_2,s_1+1
    theta2: s_3,s_2+1, where s_3 = K
    X: m,n+1 (DO include bias term in input X)
    
    Returns ...
    a1: m,n+1 (equals X)
    a2: m,s_2
    a3: m,s_3 (equals m,K)
    z2: m,s_2
    z3: m,s_3
    
    Where ...
    m = number of samples
    n = number of input features
    s_2 = number of units in layer 2 (hidden layer)
    s_3 = number of units in layer 3 (output layer)
    K = number of output classes
    """
    m = X.shape[0] # number of samples
    n = X.shape[1]-1 # number of features
    s_2 = theta1.shape[0] # number of units in layer 2
    s_1 = theta1.shape[1]-1 # number of units in layer 1
    assert s_1 == n # validate number of units in layer 1 == number of input features
    K = s_3 = theta2.shape[0] # number of output classes K == number of units in layer 3
    assert s_2 == theta2.shape[1]-1 # more input validation
    L = 3 # number of layers
    
    bias = np.ones((m,1))
    a1 = X
    z2 = X.dot(theta1.transpose())
    a2 = sigmoid(z2)
    z3 = np.hstack((bias,a2)).dot(theta2.transpose())
    a3 = sigmoid(z3)
    
    return a1,a2,a3,z2,z3
    