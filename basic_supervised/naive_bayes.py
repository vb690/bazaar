import math

import numpy as np

class BernulliNaiveBayes():

    def __init__(self):
        self.y_predicted = []

    @staticmethod
    def group_by(values_array, grouping_array):
        grouped = {group : [] for group in np.unique(grouping_array)}
        for values, group in zip(values_array, grouping_array):

            grouped[group].append(values)

        grouped = {group : np.array(values) for group, values in grouped.items()}
        return grouped

    @staticmethod
    def compute_bernulli_pdf(x, probability):
        if x == 1:
            return probability
        else:
            return 1-probability

    def _compute_labels_probabilities(self):
        labels, occurences = np.unique(self.y_train, return_counts=True)
        return {label : occurence/len(self.y_train) for label, occurence in zip(labels, occurences)}

    def _compute_features_probabilities(self, grouped):
        return {label : np.sum(values, axis = 0) / len(values) for label, values in grouped.items()}

    def _compute_likelihoods(self, input_X):
        labels_likelihoods = {}
        for label, probabilities in self.features_probabilities.items():

            likelihood = np.prod([self.compute_bernulli_pdf(x, probability) for x, probability in zip(input_X, probabilities)])
            labels_likelihoods[label] = likelihood

        return labels_likelihoods

    def _compute_joint_probabilities(self, input_X):
        labels_joint_probabilities = {}
        labels_likelihoods = self._compute_likelihoods(input_X)
        for label, probability in self.labels_probabilities.items():

            labels_joint_probabilities[label] = probability*labels_likelihoods[label]

        return labels_joint_probabilities

    def fit(self, X_train, y_train, labels_probabilities = None):
        setattr(self, 'X_train', X_train)
        setattr(self, 'y_train', y_train)
        if labels_probabilities != None:
            setattr(self, 'labels_probabilities', labels_probabilities)
        else:
            setattr(self, 'labels_probabilities', self._compute_labels_probabilities())
        grouped = self.group_by(self.X_train, self.y_train)
        setattr(self, 'features_probabilities', self._compute_features_probabilities(grouped))

    def predict(self, X_test):
        for input_X in X_test:

            labels_joint_probabilities = self._compute_joint_probabilities(input_X)
            marginal_probability = sum(labels_joint_probabilities.values())
            prediction = {}
            for label, joint_probability in labels_joint_probabilities.items():

                prediction[label] = joint_probability/marginal_probability

            self.y_predicted.append(max(prediction, key=prediction.get))

        return self.y_predicted

class GaussianNaiveBayes():

    def __init__(self):
        self.y_predicted = []

    @staticmethod
    def group_by(values_array, grouping_array):
        grouped = {group : [] for group in np.unique(grouping_array)}
        for values, group in zip(values_array, grouping_array):

            grouped[group].append(values)

        grouped = {group : np.array(values) for group, values in grouped.items()}
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

            labels_stats[label] = {'means' : values.mean(axis=0), 'stds' : values.std(axis=0)}

        return labels_stats

    def _compute_probabilities(self):
        labels, occurences = np.unique(self.y_train, return_counts=True)
        return {label : occurence/len(self.y_train) for label, occurence in zip(labels, occurences)}

    def _compute_likelihoods(self, input_X):
        labels_likelihoods = {}
        for label, stats in self.labels_stats.items():

            means = stats['means']
            stds = stats['stds']
            likelihood = np.prod([self.compute_gaussian_pdf(x, mean, std) for x, mean, std in zip(input_X, means, stds)])
            labels_likelihoods[label] = likelihood

        return labels_likelihoods

    def _compute_joint_probabilities(self, input_X):
        labels_joint_probabilities = {}
        labels_likelihoods = self._compute_likelihoods(input_X)
        for label, probability in self.labels_probabilities.items():

            labels_joint_probabilities[label] = probability*labels_likelihoods[label]

        return labels_joint_probabilities

    def fit(self, X_train, y_train, labels_probabilities = None):
        setattr(self, 'X_train', X_train)
        setattr(self, 'y_train', y_train)
        if labels_probabilities != None:
            setattr(self, 'labels_probabilities', labels_probabilities)
        else:
            setattr(self, 'labels_probabilities', self._compute_probabilities())
        grouped = self.group_by(self.X_train, self.y_train)
        setattr(self, 'labels_stats', self._compute_statistics(grouped))

    def predict(self, X_test):
        for input_X in X_test:

            labels_joint_probabilities = self._compute_joint_probabilities(input_X)
            marginal_probability = sum(labels_joint_probabilities.values())
            prediction = {}
            for label, joint_probability in labels_joint_probabilities.items():

                prediction[label] = joint_probability/marginal_probability

            self.y_predicted.append(max(prediction, key=prediction.get))

        return self.y_predicted
