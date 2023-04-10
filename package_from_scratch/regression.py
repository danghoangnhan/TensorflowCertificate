import numpy as np


class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000,weights = None,bias=0):
        self.learningrate = lr
        self.iteration = n_iters
        self.weights = weights
        self.bias = bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # self.weights = np.zeros(n_features)
        self.weights = np.array([0., 1.]).reshape(-1, 1)


        cost = np.zeros((self.iteration, 1))
        for i in range(1, self.iteration):
            r = self.predict(X) - y
            cost[i] = 0.5 * np.sum(r * r)
            self.weights[0] -= self.learningrate * np.sum(r)
            # correct the shape dimension
            self.weights[1] -= self.learningrate * np.sum(np.multiply(r, X[:, 1].reshape(-1, 1)))
            print(cost[i])

    def predict(self, X):
        return np.dot(X, self.weights)+self.bias


