import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, K=2, max_iterations=100):
        self.K = K
        self.max_iterations = max_iterations

    def make_clusters(self, X, centroids):
        clusters = [ [] for _ in range(self.K)]
        X = np.array(X) #transform from pd dataframe to numpy ndarray
        for point_i, point_i_pos in enumerate(X):
            if point_i > 0:
                closest_centroid = np.argmin(euclidean_distance(point_i_pos, centroids))
                clusters[closest_centroid].append(point_i)
        return clusters

    def improve_centroids(self, X, clusters):
        new_centroids = np.zeros(shape=(self.K, self.n))
        for i, cluster in enumerate(clusters):
            new_centroid = np.mean(X.to_numpy()[cluster], axis=0)
            new_centroids[i] = new_centroid
        
        return new_centroids

    def fit(self, X, preprocess=False):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        self.preprocess = preprocess
        if self.preprocess:
            X = preprocess_data(X)
        #m is number of samples and n is number of features
        self.m, self.n = X.shape
        self.centroids = self.init_centroidsPP(X)
        self.clusters = self.make_clusters(X, self.centroids)
        
        for i in range(self.max_iterations):
            self.centroids = self.improve_centroids(X,self.clusters)
            old_clusters = list(self.clusters)
            self.clusters = self.make_clusters(X,self.centroids)
            new_clusters = list(self.clusters)

            if (old_clusters==new_clusters):
                print(f"""
                achieved best possible result after {i} iterations
                """)
                break
                  
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        y_pred = np.zeros(self.m)
        clusters = self.make_clusters(X,self.centroids)
        for cluster_index, cluster in enumerate(clusters):
            for sample in cluster:
                y_pred[sample] = cluster_index
        returnArray = []
        for i in range(len(y_pred)):
            returnArray.append(int(y_pred[i]))
        
        return returnArray

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids
    
    def init_centroidsPP(self,X):
        centroids = np.zeros(shape=(self.K, self.n))
        #pick any point for the first centroid
        centroids[0] = X.sample(replace = False).any() 
        
        for centroid_i in range(1, self.K):
            distances = np.zeros(len(X.values))
            for sample_no, sample in enumerate(X.values):
                min_dist = np.array([None, float('inf')])
                for i in range(centroid_i):
                    centroid = centroids[i]
                    new_dist = euclidean_distance(sample, centroid)
                    if new_dist < min_dist[1]: 
                            min_dist = [i, new_dist]
                distances[sample_no] = min_dist[1]
            w = np.square(distances)/np.sum(np.square(distances))
            new_centroid = np.random.choice(self.m, p = w)
            centroids[centroid_i] = X.values[new_centroid]
        
        return centroids
    

    
# --- Some utility functions 

def preprocess_data(X):
    #X = np.array(X)
    X['x1'] = X['x1']*10
    return X

def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])

def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
