import unittest
import os
import numpy as np
from scipy.io import loadmat

from utils import cost_function_loop,regularized_cost_function_loop
from utils import sigmoid_gradient,generate_theta
from utils import nnCostFunction, unrolltheta, nnTrain, nnPredict, accuracy
from gradientcheck import checkNNGradients

DATA_DIR = os.path.join(os.path.dirname(__file__),os.pardir,'data')

class NeuralNetTestCase(unittest.TestCase):
    
    def setUp(self):
        self.data = loadmat( os.path.join(DATA_DIR,'ex4data1.mat') )
        self.X = self.data['X']
        self.m = self.X.shape[0] # number of samples
        self.n = self.X.shape[1] # number of features
        self.orig_y = self.data['y']
        self.K = np.unique(self.orig_y).size # number of output classes
        
        # vectorize y
        self.y = np.zeros((self.m, self.K))
        for i in range(0,self.m):
            self.y[i,self.orig_y[i]-1]=1
        
        self.weights = loadmat( os.path.join(DATA_DIR, 'ex4weights.mat') )
        self.theta1 = self.weights['Theta1']
        self.theta2 = self.weights['Theta2']
        self.s_1 = self.theta1.shape[1]-1 # number of units in layer 1 (excluding bias unit)
        self.s_2 = self.theta1.shape[0] # number of units in layer 2 (excluding bias unit)
        self.s_3 = self.theta2.shape[0] # number of units in layer 3
        
        # some input validation
        self.assertEqual(self.s_3, self.K)
        self.assertEqual(self.theta2.shape[1]-1, self.s_2)
        
    def test_cost_function(self):
        X = np.hstack((np.ones((self.m,1)), self.X))
        cost = cost_function_loop(self.theta1, self.theta2, X, self.y)
        self.assertAlmostEqual(cost, 0.287629, places=6)
    
    def test_regularized_cost_function(self):
        X = np.hstack((np.ones((self.m,1)), self.X))
        lamda = 1.0
        cost = regularized_cost_function_loop(self.theta1, self.theta2, X, self.y, lamda)
        self.assertAlmostEqual(cost, 0.383770, places=6)
    
    def test_nnCostFunction(self):
        cost = nnCostFunction(unrolltheta(self.theta1, self.theta2), 
                              self.s_1, self.s_2, self.K, 
                              self.X, self.y, lamda=0)
        self.assertAlmostEqual(cost, 0.287629, places=6)
        
        cost_reg = nnCostFunction(unrolltheta(self.theta1, self.theta2),
                                  self.s_1, self.s_2, self.K,
                                  self.X, self.y, lamda=1.0)
        self.assertAlmostEqual(cost_reg, 0.383770, places=6)
    
    def test_sigmoid_gradient(self):
        np.testing.assert_equal(sigmoid_gradient(np.zeros(1)), np.repeat(0.25,1))
        np.testing.assert_equal(sigmoid_gradient(np.zeros(3)), np.repeat(0.25,3))
        np.testing.assert_equal(sigmoid_gradient(np.zeros((3,3))), np.repeat(0.25,9).reshape(3,3))
    
    def test_nnGradientFunction(self):
        """
        checkNNGradient calls nnGradientFunction with a small sample neural network
        then compares the gradients returned by nnGradientFunction, with the numerical gradient
        assert the two gradients agree to 9 decimal points.
        """
        lamda = 0.0
        gradients, numgrad = checkNNGradients(lamda)
        np.testing.assert_almost_equal(gradients, numgrad, decimal=9)
        
        lamda = 3.0 # now check it with regularization term
        gradients, numgrad = checkNNGradients(lamda)
        np.testing.assert_almost_equal(gradients, numgrad, decimal=9)
    
    def test_neuralnetwork(self):
        lamda = 1.0
        epsilon = 0.12
        initial_theta1 = generate_theta(self.theta1.shape, epsilon)
        initial_theta2 = generate_theta(self.theta2.shape, epsilon)
        initial_theta = unrolltheta(initial_theta1, initial_theta2)
        optimized_theta1, optimized_theta2 = nnTrain(initial_theta, self.s_1, self.s_2, self.K, self.X, self.y, lamda)
        predictions = nnPredict(optimized_theta1, optimized_theta2, self.X)
        expected = (self.orig_y - 1).reshape(-1)
        acc = accuracy(predictions,expected)
        print "Accuracy: {} (Should be around 95%, plus or minus 1% due to random initialization)".format(acc)
        self.assertGreater(acc, 93)
        
if __name__ == '__main__':
    unittest.main()