# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 01:51:18 2021

@author: user
"""


# import packages
import numpy as np


class Perceptron:
    #defines the constructor to our Perceptron class
    def __init__(self, N, eta=0.1): 
        # N: The number of columns in our input feature vectors
        # eta: learning rate for the Perceptron algorithm. We’ll set this value to 0.01 by default.
        # Common choices of learning rates are normally in the range a = 0:1;0:01;0:001.
        # initialize weight matrix and store the learning rate
        # Note: - using a "normal distribution"
        #      - N+1 (+1 for the bias param)
        #      - Divide W by the square-root of the number of inputs. Common
        #           technique for scaling our weight matrix (faster convergence)
        self.W = np.random.randn(N+1) / np.sqrt(N)
        self.eta = eta

    def step(self, x):
        # apply the step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # insert a column of 1's as the last entry in the feature matrix (bias trick)
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point:
            for (x, target) in zip(X, y):
                # take the dot product between the input features
                # and the weight matrix, then pass this value
                # through the step function to obtain the prediction 
                p = self.step(np.dot(x, self.W))

                if p != target:
                    # determine the error
                    error = p - target
                    # update the weight matrix
                    self.W += -self.eta * error * x

    def predict(self, X, addBias=True):
        # ensure our input is a matrix
        X  = np.atleast_2d(X)

        # check to see if the bias column should be added
        if addBias:
            # insert column of 1's as the last entry in the feature
            # matrix (bias)
            X = np.c_[X, np.ones((X.shape[0]))]
            # take the dot product between the input features and the
            # weight matrix, then pass the value through the step function
        return self.step(np.dot(X, self.W))


# construct the AND dataset
#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0], [0], [0], [1]])
# construct the OR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

# define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], eta=0.1)
p.fit(X, y, epochs=20)
#now that our perceptron is trained we can evaluate it
print("[INFO] testing perceptron...")

# now that our network is trained, loop over the data points
for (x, target) in zip(X, y):
    # make a prediction on the data point and display the result
    # to our console
    pred = p.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))
