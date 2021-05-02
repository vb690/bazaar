import math

import itertools

import numpy as np

import matplotlib.pyplot as plt

from .toolbox.abstract_models import _Distances


class Knn(_Distances):
    def __init__(self, k=6, distance='euclidian'):
        self.X_train = None
        self.y_train = None
        self.y_predicted = None
        self.k = k
        self.distance = distance
        self.distances_matrix = None
        self.nearest_neighbours = None
        self.votes = None

    def _compute_distance_matrix(self, input_X):
        self.distances_matrix = getattr(self, self.distance)(
            X_train=self.X_train,
            input_X=input_X
        )
        self.nearest_neighbours = self.distances_matrix.argsort()

    def _vote(self):
        self.votes = {label: 0 for label in list(set(self.y_train))}
        for neighbour in range(self.k):

            vote_from_label = self.y_train[self.nearest_neighbours[neighbour]]
            self.votes[vote_from_label] += 1

        return max(self.votes, key=self.votes.get)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        self.y_predicted = []
        for input_X in X_test:

            self._compute_distance_matrix(input_X)
            prediction = self._vote()
            self.y_predicted.append(prediction)

        return self.y_predicted


class LogReg:
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
        return 1.0 / (1 + np.exp(-w_X))

    def _set_X_y(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.X_train = np.mat(X)
        self.y_train = np.mat(y).T
        self.weights = np.random.random((X.shape[1], 1))

    def _compute_error(self, y_predicted):
        self.error = self.y_train - y_predicted

    def _update_weights(self):
        update = self.alpha * (self.X_train.T * self.error)
        self.weights += update

    def fit(self, X_train, y_train,  epochs=500):
        self._set_X_y(X_train, y_train)
        for epoch in range(epochs):

            prediction = self.sigmoid(self.X_train * self.weights)
            self._compute_error(prediction)
            self._update_weights()

    def predict(self, X_test, proba=False):
        X_test = np.insert(X_test, 0, 1, axis=1)
        self.y_predicted = self.sigmoid(np.mat(X_test) * self.weights)
        if proba:
            return self.y_predicted
        else:
            return np.around(self.y_predicted)

    def visualize_decision_function(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis=1)
        w_X = np.mat(X_test) * self.weights
        sigmoid_w_X = self.sigmoid(w_X)
        plt.figure(figsize=(10, 10))
        plt.scatter(
            self.listify_matrix(w_X),
            self.listify_matrix(sigmoid_w_X),
            s=10
        )
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

        plt.figure(figsize=(10, 10))
        plt.scatter(
            X_1,
            X_2,
            s=10,
            c=y_test
        )
        plt.plot(X_line, y_line, color='r', linestyle='--', linewidth=1)
        plt.xlabel('X 1')
        plt.ylabel('X 2')
        plt.title('Decision Boundary')
        plt.grid(True)
        plt.show()


class BernulliNaiveBayes:
    def __init__(self):
        self.y_predicted = []

    @staticmethod
    def group_by(values_array, grouping_array):
        grouped = {group: [] for group in np.unique(grouping_array)}
        for values, group in zip(values_array, grouping_array):

            grouped[group].append(values)

        grouped = {
            group: np.array(values) for group, values in grouped.items()
        }
        return grouped

    @staticmethod
    def compute_bernulli_pdf(x, probability):
        if x == 1:
            return probability
        else:
            return 1-probability

    def _compute_labels_probabilities(self):
        labels, occurences = np.unique(self.y_train, return_counts=True)
        return {
            label: occurence/len(self.y_train) for label, occurence in
            zip(labels, occurences)
        }

    def _compute_features_probabilities(self, grouped):
        return {
            label: np.sum(values, axis=0) / len(values) for label, values in
            grouped.items()
        }

    def _compute_likelihoods(self, input_X):
        labels_likelihoods = {}
        for label, probabilities in self.features_probabilities.items():

            likelihood = np.prod(
                [self.compute_bernulli_pdf(x, prob) for x, prob in
                 zip(input_X, probabilities)
                 ]
            )
            labels_likelihoods[label] = likelihood

        return labels_likelihoods

    def _compute_joint_probabilities(self, input_X):
        labels_joint_probabilities = {}
        labels_likelihoods = self._compute_likelihoods(input_X)
        for label, probability in self.labels_probabilities.items():

            labels_joint_probabilities[label] = \
                probability*labels_likelihoods[label]

        return labels_joint_probabilities

    def fit(self, X_train, y_train, labels_probabilities=None):
        setattr(self, 'X_train', X_train)
        setattr(self, 'y_train', y_train)
        if labels_probabilities is not None:
            setattr(self, 'labels_probabilities', labels_probabilities)
        else:
            setattr(
                self,
                'labels_probabilities',
                self._compute_labels_probabilities()
            )
        grouped = self.group_by(self.X_train, self.y_train)
        setattr(
            self,
            'features_probabilities',
            self._compute_features_probabilities(grouped)
        )

    def predict(self, X_test):
        for input_X in X_test:

            labels_joint_probabilities = self._compute_joint_probabilities(
                input_X
            )
            marginal_probability = sum(labels_joint_probabilities.values())
            prediction = {}
            for label, joint_probability in labels_joint_probabilities.items():

                prediction[label] = joint_probability/marginal_probability

            self.y_predicted.append(max(prediction, key=prediction.get))

        return self.y_predicted


class GaussianNaiveBayes:
    def __init__(self):
        self.y_predicted = []

    @staticmethod
    def group_by(values_array, grouping_array):
        grouped = {group: [] for group in np.unique(grouping_array)}
        for values, group in zip(values_array, grouping_array):

            grouped[group].append(values)

        grouped = {
            group: np.array(values) for group, values in grouped.items()
        }
        return grouped

    @staticmethod
    def compute_gaussian_pdf(x, mean, std):
        variance = std**2
        num = math.exp(-(x - mean)**2 / (2 * variance))
        denom = math.sqrt(2*math.pi*variance)
        return num / denom

    def _compute_statistics(self, grouped):
        labels_stats = {}
        for label, values in grouped.items():

            labels_stats[label] = {
                'means': values.mean(axis=0),
                'stds': values.std(axis=0)
            }

        return labels_stats

    def _compute_probabilities(self):
        labels, occurences = np.unique(self.y_train, return_counts=True)
        return {
            label: occurence/len(self.y_train) for label, occurence in
            zip(labels, occurences)
        }

    def _compute_likelihoods(self, input_X):
        labels_likelihoods = {}
        for label, stats in self.labels_stats.items():

            means = stats['means']
            stds = stats['stds']
            likelihood = np.prod(
                [self.compute_gaussian_pdf(x, mean, std) for x, mean, std in
                 zip(input_X, means, stds)
                 ]
            )
            labels_likelihoods[label] = likelihood

        return labels_likelihoods

    def _compute_joint_probabilities(self, input_X):
        labels_joint_probabilities = {}
        labels_likelihoods = self._compute_likelihoods(input_X)
        for label, probability in self.labels_probabilities.items():

            labels_joint_probabilities[label] = \
                probability*labels_likelihoods[label]

        return labels_joint_probabilities

    def fit(self, X_train, y_train, labels_probabilities=None):
        setattr(self, 'X_train', X_train)
        setattr(self, 'y_train', y_train)
        if labels_probabilities is not None:
            setattr(self, 'labels_probabilities', labels_probabilities)
        else:
            setattr(
                self,
                'labels_probabilities',
                self._compute_probabilities()
            )
        grouped = self.group_by(self.X_train, self.y_train)
        setattr(self, 'labels_stats', self._compute_statistics(grouped))

    def predict(self, X_test):
        for input_X in X_test:

            labels_joint_probabilities = self._compute_joint_probabilities(
                input_X
            )
            marginal_probability = sum(labels_joint_probabilities.values())
            prediction = {}
            for label, joint_probability in labels_joint_probabilities.items():

                prediction[label] = joint_probability/marginal_probability

            self.y_predicted.append(max(prediction, key=prediction.get))

        return self.y_predicted
