import unittest
import os
import numpy as np
from scipy.io import loadmat
from sklearn import svm,grid_search
import matplotlib.pyplot as plt

from utils import gaussian_kernel

DATA_DIR = os.path.join(os.path.dirname(__file__),os.pardir,'data')

class Data1TestCase(unittest.TestCase):
    def setUp(self):
        self.data = loadmat( os.path.join(DATA_DIR,'ex6data1.mat') )
        self.X = self.data['X']
        self.y = self.data['y']
        
    def test_plot_data1(self):
        positives = self.X[ (self.y == 1).reshape(-1) ]
        negatives = self.X[ (self.y == 0).reshape(-1) ]
        plt.xlim(0,4.5)
        plt.ylim(1.5,5)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.scatter(positives[:,0], positives[:,1], s=20, c='b', marker='x', label='y=1')
        plt.scatter(negatives[:,0], negatives[:,1], s=20, c='r', marker='o', label='y=0')
        plt.legend()
        plt.show()
    
    def test_linearSVC1(self):
        linear_svm = svm.SVC(C=1.0, kernel='linear')
        linear_svm.fit(self.X, self.y)
        
        # plot the decision boundary as a contour plot
        x1_min, x1_max, x1_step = np.min(self.X[:,0]), np.max(self.X[:,0]), 0.01
        x2_min, x2_max, x2_step = np.min(self.X[:,1]), np.max(self.X[:,1]), 0.01
        x1,x2 = np.meshgrid(np.arange(x1_min, x1_max, x1_step),np.arange(x2_min, x2_max, x2_step))
        
        d = np.vstack((x1.ravel(),x2.ravel())).transpose()
        z = linear_svm.predict(d)
        z = z.reshape(x1.shape)
        plt.contour(x1, x2, z, [0,1])
        
        # plot the samples
        positives = self.X[ (self.y == 1).reshape(-1) ]
        negatives = self.X[ (self.y == 0).reshape(-1) ]
        plt.xlim(0,4.5)
        plt.ylim(1.5,5)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.scatter(positives[:,0], positives[:,1], s=20, c='b', marker='x', label='y=1')
        plt.scatter(negatives[:,0], negatives[:,1], s=20, c='r', marker='o', label='y=0')
        plt.legend()
        plt.show()
    
    def test_linearSVC2(self):
        linear_svm = svm.SVC(C=100.0, kernel='linear')
        linear_svm.fit(self.X, self.y)
        
        # plot the decision boundary as a contour plot
        x1_min, x1_max, x1_step = np.min(self.X[:,0]), np.max(self.X[:,0]), 0.01
        x2_min, x2_max, x2_step = np.min(self.X[:,1]), np.max(self.X[:,1]), 0.01
        x1,x2 = np.meshgrid(np.arange(x1_min, x1_max, x1_step),np.arange(x2_min, x2_max, x2_step))
        
        d = np.vstack((x1.ravel(),x2.ravel())).transpose()
        z = linear_svm.predict(d)
        z = z.reshape(x1.shape)
        plt.contour(x1, x2, z, [0,1])
        
        # plot the samples
        positives = self.X[ (self.y == 1).reshape(-1) ]
        negatives = self.X[ (self.y == 0).reshape(-1) ]
        plt.xlim(0,4.5)
        plt.ylim(1.5,5)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.scatter(positives[:,0], positives[:,1], s=20, c='b', marker='x', label='y=1')
        plt.scatter(negatives[:,0], negatives[:,1], s=20, c='r', marker='o', label='y=0')
        plt.legend()
        plt.show()
        
    @unittest.skip("done")
    def test_gaussian_distance(self):
        sigma = 2
        x1 = np.array([1,2,1])
        x2 = np.array([0,4,-1])
        dist = gaussian_kernel(x1,x2,sigma)
        self.assertAlmostEqual(dist, 0.324652, places=6)

class Data2TestCase(unittest.TestCase):
    
    def setUp(self):
        self.data = loadmat( os.path.join(DATA_DIR,'ex6data2.mat') )
        self.X = self.data['X']
        self.y = self.data['y'].reshape(-1)
        
    def test_plotdata(self):
        positives = self.X[ (self.y == 1).reshape(-1) ]
        negatives = self.X[ (self.y == 0).reshape(-1) ]
        plt.xlim(0,1)
        plt.ylim(0.4,1)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.scatter(positives[:,0], positives[:,1], s=20, c='b', marker='x', label='y=1')
        plt.scatter(negatives[:,0], negatives[:,1], s=20, c='r', marker='o', label='y=0')
        plt.legend()
        plt.show()
        
    def test_svm(self):
        C = 1
        sigma = 0.01
        
        gaussian_svm = svm.SVC(C=C, kernel='rbf', gamma = 1.0 / sigma)
        gaussian_svm.fit(self.X, self.y)
        
        # plot the decision boundary as a contour plot
        x1_min, x1_max, x1_len = np.min(self.X[:,0]), np.max(self.X[:,0]), 100
        x2_min, x2_max, x2_len = np.min(self.X[:,1]), np.max(self.X[:,1]), 100
        x1,x2 = np.meshgrid(np.linspace(x1_min, x1_max, num=x1_len),np.linspace(x2_min, x2_max, x2_len))
        
        d = np.vstack((x1.ravel(),x2.ravel())).transpose()
        z = gaussian_svm.predict(d)
        z = z.reshape(x1.shape)
        plt.contour(x1, x2, z, [0,1])
        
        # plot the samples
        positives = self.X[ (self.y == 1).reshape(-1) ]
        negatives = self.X[ (self.y == 0).reshape(-1) ]
        plt.xlim(0,1)
        plt.ylim(0.4,1)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.scatter(positives[:,0], positives[:,1], s=20, c='b', marker='x', label='y=1')
        plt.scatter(negatives[:,0], negatives[:,1], s=20, c='r', marker='o', label='y=0')
        plt.legend()
        plt.show()
        
class Data3TestCase(unittest.TestCase):
    
    def setUp(self):
        self.data = loadmat( os.path.join(DATA_DIR,'ex6data3.mat') )
        self.X = self.data['X']
        self.y = self.data['y'].reshape(-1)
        self.Xval = self.data['Xval']
        self.yval = self.data['yval'].reshape(-1)
        
    def test_svm(self):
        params = {'kernel':['rbf'],
                  'C':[0.01,0.03,0.1,0.3,1,3,10,30],
                  'gamma':[1/0.01,1/0.03,1/0.1,1/0.3,1,1/3,1/10,1/30]}
        
        # In sklearn gridsearch, it doesn't split data into train/validation/test sets
        # It only needs two set. development set and evaluation set.
        # It splits the development set into folds and automatically does cross validation between the folds to get the best params.
        # Then evaluation set is used for evaluating performance on unseen data.
        # So it doesn't do it the way the homework question suggests.
        clf = grid_search.GridSearchCV(svm.SVC(), params)
        best_param = clf.fit(self.X, self.y).best_params_
        
        best_svm = svm.SVC(kernel='rbf',C=best_param['C'],gamma=best_param['gamma'])
        best_svm.fit(self.X, self.y)
        
        # plot the decision boundary as a contour plot
        x1_min, x1_max, x1_len = np.min(self.X[:,0]), np.max(self.X[:,0]), 100
        x2_min, x2_max, x2_len = np.min(self.X[:,1]), np.max(self.X[:,1]), 100
        x1,x2 = np.meshgrid(np.linspace(x1_min, x1_max, num=x1_len),np.linspace(x2_min, x2_max, x2_len))
        
        d = np.vstack((x1.ravel(),x2.ravel())).transpose()
        z = best_svm.predict(d)
        z = z.reshape(x1.shape)
        plt.contour(x1, x2, z, [0,1])
        
        # plot the samples
        positives = self.X[ (self.y == 1).reshape(-1) ]
        negatives = self.X[ (self.y == 0).reshape(-1) ]
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.scatter(positives[:,0], positives[:,1], s=20, c='b', marker='x', label='y=1')
        plt.scatter(negatives[:,0], negatives[:,1], s=20, c='r', marker='o', label='y=0')
        plt.legend()
        plt.show()
    
if __name__ == "__main__":
    unittest.main()