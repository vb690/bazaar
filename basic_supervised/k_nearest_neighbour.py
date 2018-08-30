import numpy as np

class knn():

    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.y_predicted = None
        self.distances_matrix = None
        self.nearest_neighbours = None
        self.votes = None


    def _compute_distance_matrix(self, input_X):
        differenc_matrix = np.tile(input_X, (self.X_train.shape[0], 1)) - self.X_train
        squared_difference_matrix = differenc_matrix**2
        squared_distances_matrix = squared_difference_matrix.sum(axis=1)
        self.distances_matrix = squared_distances_matrix ** 0.5
        self.nearest_neighbours = self.distances_matrix.argsort()

    def _vote(self):
        self.votes = {label : 0 for label in list(set(self.y_train))}
        for neighbour in range(k):

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
