import numpy as np


class _Distances:

    def __init__(self):
        pass

    def euclidian(self, X_train, input_X):
        """Compute the ecuclidean distance between X_train and input_X
        """
        differences_matrix = np.tile(input_X, (X_train.shape[0], 1)) - X_train
        squared_differences_matrix = differences_matrix**2
        squared_distances_matrix = squared_differences_matrix.sum(axis=1)
        return squared_distances_matrix ** 0.5

    def manhattan(self, X_train, input_X):
        """Compute the manhattan distance between X_train and input_X
        """
        differences_matrix = np.tile(input_X, (X_train.shape[0], 1)) - X_train
        differences_matrix = np.abs(differences_matrix)
        distances_matrix = differences_matrix.sum(axis=1)
        return distances_matrix

    def canberra(self, X_train, input_X):
        """Compute the canberra distance between X_train and input_X
        """
        differences_matrix = np.tile(input_X, (X_train.shape[0], 1)) - X_train
        differences_matrix = np.abs(differences_matrix)
        summations_matrix = np.tile(input_X, (X_train.shape[0], 1)) + X_train
        summations_matrix = np.abs(summations_matrix)
        divisions_matrix = differences_matrix / summations_matrix
        distances_matrix = divisions_matrix.sum(axis=1)
        return distances_matrix

    def chebychev(self, X_train, input_X):
        """Compute the chebychev distance between X_train and input_X
        """
        distances_matrix = np.tile(input_X, (X_train.shape[0], 1)) - X_train
        distances_matrix = np.abs(distances_matrix)
        distances_matrix = distances_matrix.max(axis=1)
        return distances_matrix
