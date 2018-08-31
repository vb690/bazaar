import itertools

import numpy as np

import matplotlib.pyplot as plt

class LogReg():
    def __init__(self, alpha=0.001):
        self.X_train = None
        self.y_train = None
        self.y_predicted = None
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

    def _set_X_y(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.X_train = np.mat(X)
        self.y_train = np.mat(y).T
        self.weights = np.random.random((X.shape[1], 1))

    def _compute_error(self, y_predicted):
        self.error = self.y_train - y_predicted

    def _update_weights(self):
        self.weights = self.weights + (self.alpha * (self.X_train.T * self.error))

    def fit(self, X_train, y_train,  epochs=500):
        self._set_X_y(X_train, y_train)
        for epoch in range(epochs):

            prediction = self.sigmoid(self.X_train * self.weights)
            self._compute_error(prediction)
            self._update_weights()

    def predict(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis=1)
        self.y_predicted = self.sigmoid(np.mat(X_test) * self.weights)
        self.y_predicted = [1 if prediction >= 0.5 else 0 for prediction in self.y_predicted]
        return self.y_predicted

    def visualize_decision_function(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis=1)
        w_X = np.mat(X_test) * self.weights
        sigmoid_w_X = self.sigmoid(w_X)
        plt.figure(figsize = (10, 10))
        plt.scatter(self.listify_matrix(w_X), self.listify_matrix(sigmoid_w_X), s=10)
        plt.xlabel('Weighted X')
        plt.ylabel('Sigmoid (Weighted X)')
        plt.title('Decision Function')
        plt.grid(True)
        plt.show()

    def visualize_decision_boundary(self, X_test, y_test):
        X_test = np.insert(X_test, 0, 1, axis=1)
        weights = np.asarray(self.weights)
        X_1 = X_test[:, 1]
        X_2 = X_test[:, 2]
        X_line = np.arange(min(X_1), max(X_1), 0.1)
        y_line = (-weights[0]-weights[1] * X_line)/weights[2]

        plt.figure(figsize = (10, 10))
        plt.scatter(X_1, X_2,  s=10, c = y_test)
        plt.plot(X_line, y_line, color='r', linestyle='--', linewidth=1)
        plt.xlabel('X 1')
        plt.ylabel('X 2')
        plt.title('Decision Boundary')
        plt.grid(True)
        plt.show()
