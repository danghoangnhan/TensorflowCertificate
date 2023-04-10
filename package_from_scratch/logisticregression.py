# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 04:02:05 2022

@author: tuyen
"""
# Importing Libraries
import pandas as pd
import numpy as np
from numpy import log, dot, e
from numpy.random import rand
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

###############################################################################
# # Importing dataset
df = pd.read_csv("purchased.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1:].values


# Logistic Regression
class LogisticRegression:

    def sigmoid(self, z): return 1 / (1 + e ** (-z))

    def cost_function(self, X, y, weights):
        z = dot(X, weights)
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))
        return -sum(predict_1 + predict_0) / len(X)

    def fit(self, X, y, epochs, lr):
        loss = []
        weights = rand(X.shape[1])
        N = len(X)

        for _ in range(epochs):
            # Gradient Descent
            y_hat = self.sigmoid(dot(X, weights))
            weights -= lr * dot(X.T, y_hat - y) / N
            # Saving Progress
            loss.append(self.cost_function(X, y, weights))

        self.weights = weights
        self.loss = loss

    def predict(self, X):
        # Predicting with sigmoid function
        z = dot(X, self.weights)
        # Returning binary result
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]


###############################################################################
def main():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    # Splitting dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1 / 3, random_state=42)
    # Predictions
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train, epochs=500, lr=0.5)
    y_pred = logreg.predict(X_test)
    # Evaluation
    print(classification_report(y_test, y_pred))
    print('-' * 55)
    print('Confusion Matrix\n')
    print(confusion_matrix(y_test, y_pred))

    plt.style.use('seaborn-whitegrid')
    plt.rcParams['figure.dpi'] = 227
    plt.rcParams['figure.figsize'] = (16, 5)
    plt.plot(logreg.loss)
    plt.title('Logistic Regression Training', fontSize=15)
    plt.xlabel('Epochs', fontSize=12)
    plt.ylabel('Loss', fontSize=12)
    plt.show()
main()


