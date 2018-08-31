import numpy as np

class _Distances():
    def euclidian(self, X_train, input_X):
        differences_matrix = np.tile(input_X, (X_train.shape[0], 1)) - X_train
        squared_differences_matrix = differences_matrix**2
        squared_distances_matrix = squared_differences_matrix.sum(axis=1)
        return squared_distances_matrix ** 0.5

    def manhattan(self, X_train, input_X):
        differences_matrix = np.abs(np.tile(input_X, (X_train.shape[0], 1)) - X_train)
        distances_matrix = differences_matrix.sum(axis=1)
        return distances_matrix

    def canberra(self, X_train, input_X):
        differences_matrix = np.abs(np.tile(input_X, (X_train.shape[0], 1)) - X_train)
        summations_matrix = np.abs(np.tile(input_X, (X_train.shape[0], 1)) + X_train)
        divisions_matrix = differences_matrix / summations_matrix
        distances_matrix = divisions_matrix.sum(axis=1)
        return distances_matrix

    def chebychev(self, X_train, input_X):
        differences_matrix = np.abs(np.tile(input_X, (X_train.shape[0], 1)) - X_train)
        distances_matrix = differences_matrix.max(axis=1)
        return distances_matrix

class Knn():
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
        self.distances_matrix = getattr(_Distances(), self.distance)(self.X_train, input_X)
        self.nearest_neighbours = self.distances_matrix.argsort()

    def _vote(self):
        self.votes = {label : 0 for label in list(set(self.y_train))}
        for neighbour in range(self.k):

            vote_from_label = self.y_train[self.nearest_neighbours[neighbour]]
            self.votes[vote_from_label] +=1

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
