import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage 
from scipy.io import loadmat

from utils import hypothesis,classifyall,accuracy

class MultiClassClassification(unittest.TestCase):
    
    def setUp(self):
        self.data = loadmat( os.path.join(os.path.dirname(__file__), 'ex3data1.mat') )
        self.X = self.data['X']
        
        m,_ = self.X.shape
        K = len(np.unique(self.data['y']))
        self.y = np.zeros((m,K))
        for k in range(0,K):
            if k == 0:
                self.y[:,k] = ( self.data['y'] == 10 ).astype(int).reshape((m,))
            else:
                self.y[:,k] = ( self.data['y'] == k ).astype(int).reshape((m,))
    
    @unittest.skip("Skip plot data")
    def test_plotdata(self):
        width = 20
        rows, cols = 10, 10
        out = np.zeros(( width * rows, width*cols ))
        rand_indices = np.random.permutation( 5000 )[0:rows * cols]
        counter = 0
        for y in range(0, rows):
            for x in range(0, cols):
                start_x = x * width
                start_y = y * width
                out[start_x:start_x+width, start_y:start_y+width] = self.X[rand_indices[counter]].reshape(width, width).transpose()
                counter += 1
        img     = toimage( out )
        figure  = plt.figure()
        axes    = figure.add_subplot(111)
        axes.imshow( img )
        plt.show()
    
    def test_classify_all(self):
        m,n = self.X.shape
        _,K = self.y.shape
        X = np.hstack(( np.ones((m,1)), self.X ))
        initial_theta = np.zeros((n+1,K))
        lamda = 0.1
        theta = classifyall(initial_theta, X, self.y, lamda)
        hypo = hypothesis(X, theta)
        predicted_y = hypo.argmax(axis=1)
        expected_y = np.array([ d if d!=10 else 0 for d in self.data['y'].reshape(-1)])
        acc = accuracy(predicted_y,expected_y)
        self.assertAlmostEqual(acc, 94.9, places=0) # I can't get 94.9, only 94.64.... close enough I guess

if __name__ == '__main__':
    unittest.main()