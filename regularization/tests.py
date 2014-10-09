import unittest
import numpy as np
import os
import matplotlib.pyplot as plt

from utils import sigmoid,cost_function_loop,cost_function,cost_gradient
from utils import gradient_descent,hypothesis,classify
from utils import map_feature,regularized_cost_function_loop,regularized_cost_function,regularized_cost_gradient
from utils import regularized_gradient_descent

class ClassificationTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Print out the range of the features, to see if we need to do feature scaling
        """
        data = np.genfromtxt( os.path.join(os.path.dirname(__file__),'ex2data1.txt'), delimiter=',')
        X = data[:,0:2] # SELECT col0,col1 FROM self.data
        print "Feature Ranges for ex2data1 (check to see if we need to feature scaling)..."
        print "x1: min={min}, max={max}".format(min=np.min(X[:,0]), max=np.max(X[:,0]))
        print "x2: min={min}, max={max}".format(min=np.min(X[:,1]), max=np.max(X[:,1]))
    
    def setUp(self):
        self.data = np.genfromtxt( os.path.join(os.path.dirname(__file__),'ex2data1.txt'), delimiter=',')
        self.X = self.data[:,0:2] # SELECT col0,col1 FROM self.data
        self.y = self.data[:,2] # SELECT col2 FROM self.data
        self.m, self.n = self.X.shape
        self.y = self.y.reshape((self.m,1))
        #self.X = np.concatenate([np.ones((self.m,1)),self.X],axis=1)
    
    @unittest.skip("This just makes a scatter plot of the data")
    def test_plotdata1(self):
        negatives = self.data[self.data[:,2]==0] # SELECT * FROM self.data WHERE col2 == 0
        positives = self.data[self.data[:,2]==1] # SELECT * FROM self.data WHERE col2 == 1
        plt.xlabel("Exam 1 score")
        plt.ylabel("Exam 2 score")
        plt.xlim([25, 115])
        plt.ylim([25, 115])
        plt.scatter( negatives[:, 0], negatives[:, 1], c='y', marker='o', s=40, linewidths=1, label="Not admitted" )
        plt.scatter( positives[:, 0], positives[:, 1], c='b', marker='+', s=40, linewidths=2, label="Admitted" )
        plt.legend()
        plt.show()
    
    def test_sigmoid(self):
        z = np.array([-100, 0 , 100])
        g = sigmoid(z)
        self.assertAlmostEqual(g[0], 0, places=1)
        self.assertAlmostEqual(g[1], 0.5, places=1)
        self.assertAlmostEqual(g[2], 1, places=1)
    
    def test_cost_function(self):
        self.X = np.concatenate([np.ones((self.m,1)),self.X],axis=1)
        theta = np.zeros((self.n+1,1))
        cost_loop = cost_function_loop(self.X, self.y, theta)
        cost = cost_function(self.X, self.y, theta)
        self.assertAlmostEqual(cost_loop, 0.693, places=3)
        self.assertAlmostEqual(cost, 0.693, places=3)
    
    def test_cost_gradient(self):
        self.X = np.concatenate([np.ones((self.m,1)),self.X],axis=1)
        theta = np.zeros((self.n+1,1))
        gradient = cost_gradient(self.X, self.y, theta)
        np.testing.assert_almost_equal(gradient, np.array([-0.1, -12.009217, -11.262842]).reshape((3,1)), decimal=3)
    
    def test_gradient_descent(self):
        self.X = np.concatenate([np.ones((self.m,1)),self.X],axis=1)
        theta = np.zeros((self.n+1,1))
        theta_optimized, min_cost = gradient_descent(self.X, self.y, theta)
        np.testing.assert_almost_equal(theta_optimized, np.array([-25.161301,0.206231,0.201471]), decimal=3)
        self.assertAlmostEqual(min_cost, 0.203, places=3)
    
    @unittest.skip("This just makes a plot for the decision boundary")
    def test_plot_boundary(self):
        negatives = self.data[self.data[:,2]==0] # SELECT * FROM self.data WHERE col2 == 0
        positives = self.data[self.data[:,2]==1] # SELECT * FROM self.data WHERE col2 == 1
        plt.xlabel("Exam 1 score")
        plt.ylabel("Exam 2 score")
        plt.xlim([25, 115])
        plt.ylim([25, 115])
        plt.scatter( negatives[:, 0], negatives[:, 1], c='y', marker='o', s=40, linewidths=1, label="Not admitted" )
        plt.scatter( positives[:, 0], positives[:, 1], c='b', marker='+', s=40, linewidths=2, label="Admitted" )
        plt.legend()
        
        self.X = np.concatenate([np.ones((self.m,1)),self.X],axis=1)
        theta = np.zeros((self.n+1,1))
        theta_optimized, _ = gradient_descent(self.X, self.y, theta)
        
        x1 = self.X[:,1]
        x2 = -( 1/theta_optimized[2] ) * ( theta_optimized[0] + theta_optimized[1]*x1 )
        plt.plot(x1,x2)
        plt.show()
    
    def test_prediction(self):
        self.X = np.concatenate([np.ones((self.m,1)),self.X],axis=1)
        theta = np.zeros((self.n+1,1))
        theta_optimized, _ = gradient_descent(self.X, self.y, theta)
        test_data = np.array([1,45,85]).reshape((1,3))
        prediction = hypothesis(test_data, theta_optimized)
        self.assertAlmostEqual(prediction, 0.776, places=3)
        self.assertEqual(classify(test_data, self.X, theta_optimized), 1)

class RegularizationTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Print out the range of the features, to see if we need to do feature scaling
        """
        data = np.genfromtxt( os.path.join(os.path.dirname(__file__),'ex2data2.txt'), delimiter=',')
        X = data[:,0:2] # SELECT col0,col1 FROM self.data
        print "Feature Ranges for ex2data2 (check to see if we need to feature scaling)..."
        print "x1: min={min}, max={max}".format(min=np.min(X[:,0]), max=np.max(X[:,0]))
        print "x2: min={min}, max={max}".format(min=np.min(X[:,1]), max=np.max(X[:,1]))
    
    def setUp(self):
        self.data = np.genfromtxt( os.path.join(os.path.dirname(__file__),'ex2data2.txt'), delimiter=',')
        self.X = self.data[:,0:2] # SELECT col0,col1 FROM self.data
        self.y = self.data[:,2] # SELECT col2 FROM self.data
        self.m, self.n = self.X.shape
        self.y = self.y.reshape((self.m,1))
        #self.X = np.concatenate([np.ones((self.m,1)),self.X],axis=1)
    
    @unittest.skip("This just makes a scatter plot of the data")
    def test_plotdata2(self):
        negatives = self.data[self.data[:,2]==0] # SELECT * FROM self.data WHERE col2 == 0
        positives = self.data[self.data[:,2]==1] # SELECT * FROM self.data WHERE col2 == 1
        plt.xlabel("Microchip Test 1")
        plt.ylabel("Microchip Test 2")
        plt.scatter( negatives[:, 0], negatives[:, 1], c='y', marker='o', s=40, linewidths=1, label="y=0" )
        plt.scatter( positives[:, 0], positives[:, 1], c='b', marker='+', s=40, linewidths=2, label="y=1" )
        plt.legend()
        plt.show()
    
    def test_map_feature(self):
        dimension = 6
        X_mapped = map_feature(self.X[:,0],self.X[:,1],dimension)
        expected_X_mapped = np.loadtxt(os.path.join(os.path.dirname(__file__),'ex2data2.mapped.txt'))
        np.testing.assert_almost_equal(X_mapped, expected_X_mapped, decimal=2)
        
    def test_regularized_cost_function(self):
        dimension = 6
        X_mapped = map_feature(self.X[:,0],self.X[:,1],dimension)
        m,n = X_mapped.shape
        X_mapped = np.hstack(( np.ones((m,1)), X_mapped ))
        theta = np.zeros((n+1,1))
        lamda = 1.0
        cost_loop = regularized_cost_function_loop(X_mapped,self.y,theta,lamda)
        cost = regularized_cost_function(X_mapped,self.y,theta,lamda)
        self.assertAlmostEqual(cost_loop, 0.69314718056, places=5)
        self.assertAlmostEqual(cost, 0.69314718056, places=5)
    
    def test_regularized_cost_gradient(self):
        dimension = 6
        X_mapped = map_feature(self.X[:,0],self.X[:,1],dimension)
        m,n = X_mapped.shape
        X_mapped = np.hstack(( np.ones((m,1)), X_mapped ))
        theta = np.zeros((n+1,1))
        lamda = 1.0
        gradient = regularized_cost_gradient(X_mapped, self.y, theta, lamda)
        expected_gradient = np.array([0.00847, 0.01879, 8e-05, 0.05034, 0.0115, 0.03766, 0.01836, 0.00732, 0.00819, 0.02348, 0.03935, 0.00224, 0.01286, 0.0031, 0.0393, 0.01997, 0.00433, 0.00339, 0.00584, 0.00448, 0.03101, 0.03103, 0.0011, 0.00632, 0.00041, 0.00727, 0.00138, 0.03879]).reshape((n+1,1))
        np.testing.assert_almost_equal(gradient, expected_gradient, decimal=5)
    
    def test_regularized_gradient_descent(self):
        dimension = 6
        X_mapped = map_feature(self.X[:,0],self.X[:,1],dimension)
        m,n = X_mapped.shape
        X_mapped = np.hstack(( np.ones((m,1)), X_mapped ))
        theta = np.zeros((n+1,1))
        lamda = 1.0
        theta_optimized, min_cost = regularized_gradient_descent(X_mapped, self.y, theta, lamda)
        expected_theta_optimized = np.array([ 
            1.27268726,  0.62557024,  1.18096643, -2.01919814, -0.91761464,
            -1.43194196,  0.12375928, -0.36513066, -0.35703386, -0.17485797,
            -1.4584374 , -0.05129691, -0.6160397 , -0.27464158, -1.19282551,
            -0.24270352, -0.20570051, -0.04499796, -0.27782728, -0.29525866,
            -0.45613268, -1.0437783 ,  0.02762813, -0.29265655,  0.01543383,
            -0.32759297, -0.14389219, -0.92460139])
        expected_min_cost = 0.5290027422883413
        np.testing.assert_almost_equal(theta_optimized, expected_theta_optimized, decimal=5)
        self.assertAlmostEqual(min_cost, expected_min_cost, places=3)
    
    @unittest.skip("This just plots the decision boundary")
    def test_plot_decision_boundary(self):
        negatives = self.data[self.data[:,2]==0] # SELECT * FROM self.data WHERE col2 == 0
        positives = self.data[self.data[:,2]==1] # SELECT * FROM self.data WHERE col2 == 1
        plt.xlabel("Microchip Test 1")
        plt.ylabel("Microchip Test 2")
        plt.scatter( negatives[:, 0], negatives[:, 1], c='y', marker='o', s=40, linewidths=1, label="y=0" )
        plt.scatter( positives[:, 0], positives[:, 1], c='b', marker='+', s=40, linewidths=2, label="y=1" )
        
        dimension = 6
        X_mapped = map_feature(self.X[:,0],self.X[:,1],dimension)
        m,n = X_mapped.shape
        X_mapped = np.hstack(( np.ones((m,1)), X_mapped ))
        theta = np.zeros((n+1,1))
        lamda = 1.0
        theta_optimized, min_cost = regularized_gradient_descent(X_mapped, self.y, theta, lamda)
        
        x1 = np.linspace( -1, 1.5, 50 )
        x2 = np.linspace( -1, 1.5, 50 )
        
        X1,X2 = np.meshgrid(x1,x2)
        hypo = np.zeros((len(x1),len(x2)))
        for i in range(0,len(x1)):
            for j in range(0,len(x2)):
                mapped = map_feature(np.array([X1[i][j]]).reshape((1,1)),
                                     np.array([X2[i][j]]).reshape((1,1)),
                                     dimension)
                mapped = np.hstack(( np.ones((1,1)), mapped ))
                hypo[i][j] = hypothesis(mapped, theta_optimized)[0]
        
        plt.contour(X1,X2,hypo,[0.5],label='Decision Boundary')
        plt.legend()
        plt.show()

if __name__  == "__main__":
    unittest.main()