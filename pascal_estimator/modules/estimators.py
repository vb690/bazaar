import numpy as np

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt


class BayesianPolyEst:
    """
    """
    def __init__(self, degree, **kwargs):
        """
        """
        self.transformer = PolynomialFeatures(degree=degree)
        self.regressor = BayesianRidge(**kwargs)

    @staticmethod
    def prediction_printer(X, mean, std):
        """
        """
        for x, m, s in zip(X, mean, std):

            print(f'Estimated Pascal for concentration {x}: {m} + / - {s}')

    def fit(self, X, y):
        """
        """
        X_poly = self.transformer.fit_transform(X)
        self.regressor.fit(
            X_poly,
            y
        )

    def predict(self, X, verbose=False):
        """
        """
        X_poly = self.transformer.fit_transform(X)
        predictions_mean, predictions_std = self.regressor.predict(
            X_poly,
            return_std=True
        )
        if verbose:
            self.prediction_printer(
                X,
                predictions_mean,
                predictions_std
            )
        return predictions_mean, predictions_std

    def validate(self, X, y):
        """
        """
        line = np.array([i for i in range(X.max() * 2)]).reshape(-1, 1)
        X_poly = self.transformer.fit_transform(line)
        plt.figure(figsize=(10, 10))
        predictions_mean, predictions_std = self.regressor.predict(
            X_poly,
            return_std=True
        )

        plt.plot(line, predictions_mean)
        plt.fill_between(
            line.flatten(),
            predictions_mean + (predictions_std * 2.5),
            predictions_mean - (predictions_std * 2.5),
            alpha=0.25
        )
        plt.scatter(X, y, c='r')
        plt.xlabel('Concentration')
        plt.ylabel('Estimated Pascal')
        plt.show()
