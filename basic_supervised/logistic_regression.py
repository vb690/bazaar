import itertools
import numpy as np
import matplotlib.pyplot as plt

class LogReg():
    def __init__(self, X, y, alpha, epochs=1000):
        self.X = np.mat(X)
        self.y = np.mat(y).transpose()
        self.alpha = alpha
        self.epochs = epochs
        self.weights = np.random.random((X.shape[1], 1))
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

    def _compute_error(self, y_predicted):
        self.error = self.y - y_predicted

    def _update_weights(self):
        self.weights = self.weights + (self.alpha * (self.X.T * self.error))

    def fit(self):
        for epoch in range(self.epochs):
            y_predicted = self.sigmoid(self.X * self.weights)

            self._compute_error(y_predicted)
            self._update_weights()

    def predict(self, X, y):
        X = np.mat(X)
        predicted = self.sigmoid(X * self.weights)
        predicted = [1 if prediction >= 0.5 else 0 for prediction in predicted]
        accuracy = self.compute_accuracy(predicted, y)
        print('Achieved accuracy: {}'.format(accuracy))

    def visualize_decision_function(self, X):
        w_X = X * self.weights
        sigmoid_w_X = self.sigmoid(w_X)
        plt.figure(figsize = (10, 10))
        plt.scatter(self.listify_matrix(w_X), self.listify_matrix(sigmoid_w_X))
        plt.xlabel('Weighted X')
        plt.ylabel('Sigmoid (Weighted X)')
        plt.show()
