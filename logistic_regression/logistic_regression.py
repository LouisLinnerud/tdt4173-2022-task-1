import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class LogisticRegression:
    
    def __init__(self, alpha=0.05, iterations=1000):
        self.alpha = alpha # alpha is learning rate
        self.iterations = iterations

    def fit(self, X ,Y,feature=False):
        self.feature = feature
        """ 
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        # m x n input (#features = n, #samples = m)
        if self.feature:
            p1 = [-3,3]
            p2 = [3,-3]
            X = add_linear_feature(X,p1, p2)
        self.m, self.n = X.shape 
        # setting initial weight and bias to zero
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        self.gradient_descend(self.iterations)
        

    def gradient_descend(self, n):
        for i in range(n):
            # Prediction
            Z = self.X.dot(self.w) + self.b # Z = w * X + b
            Y_hat = sigmoid(Z)
            # Update weight and bias
            dw = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.Y))
            db = (1 / self.m) * np.sum(Y_hat - self.Y) 
                #TODO
                #is np.sum above nessicary?
            self.w -= self.alpha * dw
            self.b -= self.alpha * db
   
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        if self.feature:
            p1 = [-3,3]
            p2 = [3,-3]
            X = add_linear_feature(X,p1,p2)
        Z = X.dot(self.w) + self.b # Z = w * X + b
        Y_hat = np.where(sigmoid(Z) > 0.5 , 1, 0)

        return Y_hat

        
    # --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    
def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )

def sigmoid( x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

def add_linear_feature(X,p1, p2):
    p1=np.array(p1)
    p2=np.array(p2)
    X = np.array(X)
    distances = []
    for i in range(0,len(X)):
        pos = X[i]
        point = np.array(pos)
        d = np.linalg.norm(np.cross(p2-p1, p1-point))/np.linalg.norm(p2-p1)
        distances.append(d)
    feature_array = []
    for i in range(len(distances)):
        feature_array.append( [X[i][0],X[i][1],distances[i]])
    feature_array = np.array(feature_array)
    return feature_array


