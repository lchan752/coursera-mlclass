import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

from utils import cost_function, gradient_function
from utils import train_linear_reg, predictions
from utils import learning_curve, map_poly_features, feature_normalization
from utils import validation_curve

DATA_DIR = os.path.join(os.path.dirname(__file__),os.pardir,'data')

class LinearRegressionTestCase(unittest.TestCase):
    
    def setUp(self):
        self.data = loadmat( os.path.join(DATA_DIR,'ex5data1.mat') )
        self.X = self.data['X']
        self.y = self.data['y']
        self.Xval = self.data['Xval']
        self.yval = self.data['yval']
        self.Xtest = self.data['Xtest']
        self.ytest = self.data['ytest']
        
        self.m , self.n = self.X.shape
        self.mval, self.nval = self.Xval.shape
        self.mtest, self.ntest = self.Xtest.shape
    
    def test_plot_traindata(self):
        plt.xlabel("Change in water level (x)")
        plt.ylabel("Water flowing out of the dam (y)")
        plt.scatter( self.X, self.y, marker='x', c='r', s=30, linewidth=2 )
        plt.show()
    
    def test_cost_function(self):
        initial_theta = np.ones(self.n+1)
        lamda = 1.0
        cost = cost_function(initial_theta, self.X, self.y, lamda)
        self.assertAlmostEqual(cost, 303.993, places=3)
        
    def test_gradient_function(self):
        initial_theta = np.ones(self.n+1)
        lamda = 1.0
        gradient = gradient_function(initial_theta, self.X, self.y, lamda)
        np.testing.assert_almost_equal(gradient, np.array([-15.30, 598.250]), decimal=2)
        
    def test_linear_regression(self):
        initial_theta = np.ones(self.n+1)
        lamda = 0.0
        theta = train_linear_reg(initial_theta, self.X, self.y, lamda)
        hypo = predictions(theta, self.X)
        
        # now scatter the data and plot the hypothesis
        plt.xlabel("Change in water level (x)")
        plt.ylabel("Water flowing out of the dam (y)")
        plt.scatter( self.X, self.y, marker='x', c='r', s=30, linewidth=2 )
        plt.plot( self.X, hypo )
        plt.show()
    
    def test_learning_curve(self):
        error_train, error_val = learning_curve(self.X, self.y, self.Xval, self.yval, lamda=0.0)
        plt.xlabel('Number of Training Examples')
        plt.ylabel("Error")
        plt.plot( range(0,self.m), error_train)
        plt.plot( range(0,self.m), error_val)
        plt.show()
        
    def test_linear_regression_poly(self):
        highest_degree = 8
        X_poly = map_poly_features(self.X, highest_degree)
        X_poly, mu, sigma = feature_normalization(X_poly)
        
        initial_theta = np.ones(highest_degree+1)
        lamda = 0.0
        theta = train_linear_reg(initial_theta, X_poly, self.y, lamda)
        hypo = predictions(theta, X_poly)
        
        # now scatter the data and plot the hypothesis
        df = pd.DataFrame(np.hstack(( self.X, hypo.reshape(self.y.shape), self.y )), columns=['X','hypo','y'])
        df = df.sort('X')
        plt.xlabel("Change in water level (x)")
        plt.ylabel("Water flowing out of the dam (y)")
        plt.scatter( df['X'], df['y'], marker='x', c='r', s=30, linewidth=2 )
        plt.plot( df['X'], df['hypo'], linestyle='--', linewidth=3 )
        plt.show()
    
    def test_learning_curve_poly(self):
        highest_degree = 8
        X_poly = map_poly_features(self.X, highest_degree)
        X_poly, mu, sigma = feature_normalization(X_poly)
        Xval_poly = map_poly_features(self.Xval, highest_degree)
        Xval_poly = (Xval_poly - mu) / sigma
        lamda = 0.0
        
        error_train, error_val = learning_curve(X_poly, self.y, Xval_poly, self.yval, lamda)
        plt.xlabel('Number of Training Examples')
        plt.ylabel("Error")
        plt.plot( range(1,self.m+1), error_train)
        plt.plot( range(1,self.m+1), error_val)
        plt.show()
        
    def test_validation_curve_poly(self):
        highest_degree = 8
        X_poly = map_poly_features(self.X, highest_degree)
        X_poly, mu, sigma = feature_normalization(X_poly)
        Xval_poly = map_poly_features(self.Xval, highest_degree)
        Xval_poly = (Xval_poly - mu) / sigma
        lamdas = np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10])
        
        error_train, error_val = validation_curve(X_poly, self.y, Xval_poly, self.yval, lamdas)
        plt.xlabel('lamda')
        plt.ylabel("Error")
        plt.plot( lamdas, error_train, label='Train')
        plt.plot( lamdas, error_val, label='Validation')
        plt.legend()
        plt.show()
    
if __name__ == "__main__":
    unittest.main()