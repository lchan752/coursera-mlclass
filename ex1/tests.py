import unittest
from utils import hypothesis,computeCost,computeCostLoop,gradientDescent
from utils import feature_normalization_loop, feature_normalization, normal_equation

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__),os.pardir,'data')

class LinearRegressionTestCase(unittest.TestCase):
    
    def setUp(self):
        self.data = np.genfromtxt( os.path.join(DATA_DIR,'ex1data1.txt'), delimiter=',')
        self.X, self.y = self.data[:, 0], self.data[:, 1]
        self.m = len(self.y)
        self.X = self.X.reshape(self.m,1)
        self.y = self.y.reshape(self.m, 1)
        self.X = np.concatenate([np.ones((self.m,1)),self.X],axis=1)
    
    def test_cost(self):
        theta = np.zeros((2,1))
        self.assertAlmostEqual(computeCostLoop(self.X, self.y, theta), 32.07, places=2)
        self.assertAlmostEqual(computeCost(self.X, self.y, theta), 32.07, places=2)
    
    def test_gradientDescent(self):
        theta = np.zeros((2,1))
        iterations = 1500
        alpha = 0.01
        converged_theta = gradientDescent(self.X, self.y, theta, alpha, iterations)
        self.assertAlmostEqual(converged_theta[0], -3.63, places=2)
        self.assertAlmostEqual(converged_theta[1], 1.17, places=2)
        self.assertAlmostEqual(hypothesis(np.array([1,3.5]),converged_theta), 0.45, places=2)
        self.assertAlmostEqual(hypothesis(np.array([1,7]),converged_theta), 4.53, places=2)

class LinearRegressionMultiVariableTestCase(unittest.TestCase):
    
    def setUp(self):
        self.data = np.genfromtxt( os.path.join(DATA_DIR,'ex1data2.txt'), delimiter=',')
        self.X, self.y = self.data[:, 0:2], self.data[:, 2]
        self.m = len(self.y)
        self.X = self.X.reshape(self.m, 2)
        self.y = self.y.reshape(self.m, 1)
    
    def test_feature_normalization(self):
        X_norm, mu, sigma = feature_normalization(self.X)
        X_norm_loop, mu_loop, sigma_loop = feature_normalization_loop(self.X)
        
        expected_X_norm = np.array([
            [ 0.13, -0.22],
            [-0.5,  -0.22],
            [ 0.5,  -0.22],
            [-0.74, -1.54],
            [ 1.26,  1.09],
            [-0.02,  1.09],
            [-0.59, -0.22],
            [-0.72, -0.22],
            [-0.78, -0.22],
            [-0.64, -0.22],
            [-0.08,  1.09],
            [-0.0,  -0.22],
            [-0.14, -0.22],
            [ 3.12,  2.4 ],
            [-0.92, -0.22],
            [ 0.38,  1.09],
            [-0.86, -1.54],
            [-0.96, -0.22],
            [ 0.77,  1.09],
            [ 1.3,   1.09],
            [-0.29, -0.22],
            [-0.14, -1.54],
            [-0.5,  -0.22],
            [-0.05,  1.09],
            [ 2.38, -0.22],
            [-1.13, -0.22],
            [-0.68, -0.22],
            [ 0.66, -0.22],
            [ 0.25, -0.22],
            [ 0.8,  -0.22],
            [-0.2,  -1.54],
            [-1.26, -2.85],
            [ 0.05,  1.09],
            [ 1.43, -0.22],
            [-0.24,  1.09],
            [-0.71, -0.22],
            [-0.96, -0.22],
            [ 0.17,  1.09],
            [ 2.79,  1.09],
            [ 0.2,   1.09],
            [-0.42, -1.54],
            [ 0.3,  -0.22],
            [ 0.71,  1.09],
            [-1.01, -0.22],
            [-1.45, -1.54],
            [-0.19,  1.09],
            [-1.0,   -0.22],
        ])
        expected_mu = np.array([2000.68,3.17]).reshape((2,1))
        expected_sigma = np.array([794.7,0.76]).reshape((2,1))
        
        np.testing.assert_almost_equal(X_norm_loop, expected_X_norm, decimal=2)
        np.testing.assert_almost_equal(mu_loop, expected_mu, decimal=2)
        np.testing.assert_almost_equal(sigma_loop, expected_sigma, decimal=2)
        
        np.testing.assert_almost_equal(X_norm, expected_X_norm, decimal=2)
        np.testing.assert_almost_equal(mu, expected_mu, decimal=2)
        np.testing.assert_almost_equal(sigma, expected_sigma, decimal=2)
    
    def test_gradient_descent(self):
        iterations = 400
        alpha = 0.01 #[0.01, 0.03, 0.1, 0.3, 1.0]
        m,n = self.X.shape
        theta = np.zeros((n+1,1))
        
        # add x_0 and do feature normalization on the rest of the columns
        self.X = np.concatenate([np.ones((self.m,1)),self.X],axis=1)
        self.X[:,1:n+1], mu, sigma = feature_normalization(self.X[:,1:n+1])
        
        theta = gradientDescent(self.X, self.y, theta, alpha, iterations)
        
        test = np.array([1.0, 1650.0, 3.0]).reshape((3,1))
        test[1:,:] = ( test[1:,:] - mu ) / sigma
        test = test.reshape((1,3)) # m=1, n=2, because there is 1 test case, with 2 features in it
        self.assertAlmostEqual(hypothesis(test, theta), 289314.62, places=2)
    
    def test_normal_quation(self):
        self.X = np.concatenate([np.ones((self.m,1)),self.X],axis=1)
        theta = normal_equation(self.X, self.y)
        test = np.array([1.0, 1650.0, 3.0]).reshape((1,3))
        self.assertAlmostEqual(hypothesis(test, theta), 293081.46, places=2)

if __name__ == '__main__':
    unittest.main()