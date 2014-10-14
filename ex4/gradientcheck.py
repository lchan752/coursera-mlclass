import numpy as np

from utils import nnCostFunction, nnGradientFunction

def debugInitializeWeights(fan_out, fan_in):
    num_elements = fan_out * (1+fan_in)
    w = np.array([np.sin(x) / 10 for x in range(1, num_elements+1)])
    w = w.reshape( 1+fan_in, fan_out ).T
    return w

def computeNumericalGradient( theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamda ):
    numgrad     = np.zeros( np.shape(theta) )
    perturb     = np.zeros( np.shape(theta) ) #38 x 1
    e = 1e-4

    num_elements = np.shape(theta)[0]

    for p in range(0, num_elements) :
        perturb[p] = e
        loss1 = nnCostFunction( theta - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)
        loss2 = nnCostFunction( theta + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad

def checkNNGradients( lamda = 0.0 ):
    input_layer_size     = 3
    hidden_layer_size     = 5
    num_labels             = 3
    m                     = 5

    theta1 = debugInitializeWeights( hidden_layer_size, input_layer_size )
    theta2 = debugInitializeWeights( num_labels, hidden_layer_size )

    X = debugInitializeWeights( m, input_layer_size - 1 )
    y = np.array([
            [1,0,0],
            [1,0,0],
            [1,0,0],
            [0,1,0],
            [0,0,1],
        ])

    nn_params     = np.array([theta1.T.reshape(-1).tolist() + theta2.T.reshape(-1).tolist()]).T
    gradient     = nnGradientFunction( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda )
    numgrad     = computeNumericalGradient( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda )
    return gradient, numgrad.reshape(-1)