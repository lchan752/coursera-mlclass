import unittest
import os
import numpy as np
from scipy.io import loadmat
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils import find_closest_centroids, compute_centroids
from utils import kmeans, kmeans_without_K

DATA_DIR = os.path.join(os.path.dirname(__file__),os.pardir,'data')

class KMeansTestCase(unittest.TestCase):
    
    def setUp(self):
        self.data = loadmat( os.path.join(DATA_DIR,'ex7data2.mat') )
        self.X = self.data['X']
    
    @unittest.skip("done")
    def test_find_closest_centroids(self):
        initial_centroids = np.array([[3,3],
                                      [6,2],
                                      [8,5]])
        idx = find_closest_centroids(self.X[0:3,:], initial_centroids)
        np.testing.assert_equal(idx, np.array([1,3,2]))
    
    @unittest.skip("done")
    def test_compute_centroids(self):
        initial_centroids = np.array([[3,3],
                                      [6,2],
                                      [8,5]])
        idx = find_closest_centroids(self.X, initial_centroids)
        centroids = compute_centroids(self.X, idx, initial_centroids.shape[0])
        expected_centroids = np.array([[2.428301, 3.157924],
                                       [5.813503, 2.633656],
                                       [7.119387, 3.616684]])
        np.testing.assert_almost_equal(centroids, expected_centroids, decimal=6)
    
    @unittest.skip("done")
    def test_kmeans(self):
        K = 3
        colors = ['r','b','g']
        labels = kmeans(self.X, K)
        
        for l in range(0,K):
            # make scatter plot for cluster l
            x1 = self.X[labels == l][:,0]
            x2 = self.X[labels == l][:,1]
            plt.ylim(0,6)
            plt.xlim(-1,9)
            plt.scatter(x1, x2, c=colors[l], marker='o')
        
        plt.show()
    
    @unittest.skip("done")
    def test_kmeans2(self):
        """
        Not a homework problem. Just trying things out on sklearn
        """
        labels = kmeans_without_K(self.X)
        labels_true = kmeans(self.X, 3)
        
        print 'adjusted rand score: {}'.format(metrics.adjusted_rand_score(labels_true, labels))
        
        K = np.unique(labels).size
        colors = cm.rainbow(np.linspace(0,1,K))
        
        for l in range(0,K):
            # make scatter plot for cluster l
            x1 = self.X[labels == l][:,0]
            x2 = self.X[labels == l][:,1]
            plt.ylim(0,6)
            plt.xlim(-1,9)
            plt.scatter(x1, x2, c=colors[l], marker='o')
        
        plt.show()

if __name__ == "__main__":
    unittest.main()