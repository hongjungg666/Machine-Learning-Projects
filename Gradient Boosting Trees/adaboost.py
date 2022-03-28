import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    ### BEGIN SOLUTION
    data = np.genfromtxt(filename,delimiter=",", dtype = float)
    Y = data[:,-1]
    X = data[:,:-1]  
    my_dict={0:-1,1:1}
    Y = np.vectorize(my_dict.get)(Y)
    ### END SOLUTION
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 

    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    tree weight: The effect of the stage weight is that more accurate models have more weight or 
    contribution to the final prediction.
    point weight:This has the effect of not changing the weight if the training instance was classified correctly 
    and making the weight slightly larger if the weak learner misclassified the instance.
    """
    trees = []
    trees_weights = []
    N, _ = X.shape
    d = np.ones(N) / N  # initial weights
    # BEGIN SOLUTION
    for i in range(num_iter):
        stump = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        stump.fit(X, y, sample_weight=d)
        terror = np.abs(stump.predict(X)-y)/2  # if it's not 0 will divide others 0
        error = np.sum(d*terror)/np.sum(d)
        stage = np.log((1-error) / (error+0.0000000001))
        d = d * np.exp(stage * terror)
        trees_weights.append(stage)
        trees.append(stump)
    # END SOLUTION
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
    """
    # X input, y output
    N, _ =  X.shape
    y = np.zeros(N)
    ### BEGIN SOLUTION
    for i in range(len(trees)):
        y = y + trees[i].predict(X)*trees_weights[i]
    y = np.sign(y)
    ### END SOLUTION
    return y
