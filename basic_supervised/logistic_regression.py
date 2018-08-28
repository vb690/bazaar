import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from sklearn.model_selection import StratifiedShuffleSplit

class LogReg():
    def __init__(self, alpha=0.001):
        self.X = None
        self.y = None
        self.alpha = alpha
        self.weights = None
        self.error = None

    @staticmethod
    def listify_matrix(matrix):
        matrix = np.array(matrix).tolist()
        return list(itertools.chain.from_iterable(matrix))

    @staticmethod
    def sigmoid(w_X):
        return 1.0 / (1+ np.exp(-w_X))

    @staticmethod
    def compute_accuracy(predicted, ground_truth):
        n_correct = 0
        for prediction, truth in zip(predicted, ground_truth):

            if prediction == truth:
                n_correct += 1

        return (n_correct/len(ground_truth))*100

    def _set_X_y(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.X = np.mat(X)
        self.y = np.mat(y).T
        self.weights = np.random.random((X.shape[1], 1))

    def _compute_error(self, y_predicted):
        self.error = self.y - y_predicted

    def _update_weights(self):
        self.weights = self.weights + (self.alpha * (self.X.T * self.error))

    def fit(self, X, y,  epochs=10):
        self._set_X_y(X, y)
        for epoch in range(epochs):
            y_predicted = self.sigmoid(self.X * self.weights)
            self._compute_error(y_predicted)
            self._update_weights()

    def predict(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        predicted = self.sigmoid(np.mat(X) * self.weights)
        predicted = [1 if prediction >= 0.5 else 0 for prediction in predicted]
        accuracy = self.compute_accuracy(predicted, y)
        print('Achieved accuracy: {}'.format(accuracy))

    def visualize_decision_function(self, X):
        X = np.insert(X, 0, 1, axis=1)
        w_X = np.mat(X) * self.weights
        sigmoid_w_X = self.sigmoid(w_X)
        plt.figure(figsize = (10, 10))
        plt.scatter(self.listify_matrix(w_X), self.listify_matrix(sigmoid_w_X), s=10)
        plt.xlabel('Weighted X')
        plt.ylabel('Sigmoid (Weighted X)')
        plt.show()

    def visualize_decision_boundary(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        weights = np.asarray(self.weights)
        X_1 = X[:, 1]
        X_2 = X[:, 2]
        X_line = np.arange(min(X_1), max(X_1), 0.1)
        y_line = (-weights[0]-weights[1] * X_line)/weights[2]

        plt.figure(figsize = (10, 10))
        plt.scatter(X_1, X_2,  s=10, c = y)
        plt.plot(X_line, y_line, color='r', linestyle='--', linewidth=1)
        plt.xlabel('X 0')
        plt.ylabel('X 1')
        plt.show()
