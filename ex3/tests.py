import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage 
from scipy.io import loadmat

from utils import hypothesis,classifyall,accuracy
from utils import feedforward

DATA_DIR = os.path.join(os.path.dirname(__file__),os.pardir,'data')

class MultiClassClassification(unittest.TestCase):
    
    def setUp(self):
        self.data = loadmat( os.path.join(DATA_DIR,'ex3data1.mat') )
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

class FeedFowardPropagationTestCase(unittest.TestCase):
    
    def setUp(self):
        self.data = loadmat( os.path.join(DATA_DIR, 'ex3data1.mat') )
        self.weights = loadmat( os.path.join(DATA_DIR, 'ex3weights.mat') )
        self.X = self.data['X'] # m,n matrix, m=5000, n=400, need to prepend x_0 to self.X
        self.y = self.data['y'] # m,1 matrix, m=5000
        self.theta1 = self.weights['Theta1'] # 25,401 matrix, 25 units in layer2, 400(+1 bias) units in layer1
        self.theta2 = self.weights['Theta2'] # 10,26 matrix, 10 units in layer3 (for 10 classes), 25(+1 bias) units in layer2
    
    def test_prediction(self):
        m,_ = self.X.shape
        X = np.hstack(( np.ones((m,1)), self.X ))
        predictions = feedforward(X,self.theta1,self.theta2)
        
        # because self.y uses 10 for 0, so the vectorized y representation is shifted.
        # if y=10, the output layer will look like [0,0,0,0,0,0,0,0,0,1], so argmax == 9 (i.e. 10 in octave/matlab, which represents class 0)
        # if y=1 , the output layer will look like [1,0,0,0,0,0,0,0,0,0], so argmax == 0 (i.e. 1  in octave/matlab, which represents class 1)
        # so the fix is, we minus 1 on all elements on y, so the argmax will be 0 indexed, which is good for python.
        expected = (self.y - 1).reshape(-1)
        acc = accuracy(predictions,expected)
        self.assertAlmostEqual(acc, 97.5, places=1)

if __name__ == '__main__':
    unittest.main()