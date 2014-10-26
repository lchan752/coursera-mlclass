import numpy as np
from sklearn import cluster
from sklearn import grid_search

def find_closest_centroids(X, centroids):
    """
    Return index of closest centroids for m samples
    
    Given ...
    X: m,n matrix
    centroids: K,n matrix
    
    Return ...
    idx: 1d array of size m, each element takes values between 1-K
    
    Where ...
    m: number of samples
    n: number of features
    K: number of clusters/centroids
    """
    
    assert X.shape[1] == centroids.shape[1]
    
    m,_ = X.shape
    K,_ = centroids.shape
    idx = np.zeros(m)
    
    for i in range(0,m):
        cost = X[i,:] - centroids
        cost = cost ** 2
        idx[i] = np.argmin(np.sum(cost, axis=1))+1
    
    return idx

def compute_centroids(X,idx,K):
    """
    computes new centroids
    
    Given ...
    X: m,n matrix
    idx: 1d array of size m, each element takes values between 1-K
    
    Return ...
    centroids: K,n matrix
    
    Where ...
    m: number of samples
    n: number of features
    K: number of clusters/centroids
    """
    m,n = X.shape
    centroids = np.zeros((K,n))
    
    for k in range(1,K+1):
        xk = X[idx == k]
        centroids[k-1] = np.mean(xk,axis=0)
        
    return centroids

def kmeans(X,K):
    clt = cluster.KMeans(n_clusters=K)
    clt.fit(X)
    return clt.labels_

def kmeans_without_K(X):
    params = [{'n_clusters':[1,2,3,4,5]}]
    clt = grid_search.GridSearchCV(estimator=cluster.KMeans(), param_grid=params, cv=5)
    clt.fit(X)
    return clt.best_estimator_.labels_
    