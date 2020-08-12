import numpy as np

import pandas as pd

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt


class BayesianPolyEst:
    """
    """
    def __init__(self, degree, sim_size=200, **kwargs):
        """
        """
        self.transformer = PolynomialFeatures(degree=degree)
        self.regressor = BayesianRidge(**kwargs)
        self.sim_size = sim_size

    @staticmethod
    def prediction_printer(X, mean, std):
        """
        """
        for x, m, s in zip(X, mean, std):

            print(f'Estimated Pascal for concentration {x}: {m} + / - {s}')

    def fit(self, X, y, simulate=False, save_name=None):
        """
        """
        setattr(self, 'save_name', save_name)
        if simulate:
            X = np.array([[x] * self.sim_size for x in X])
            X = X.reshape(-1, 1)
            y = np.array([np.random.normal(m, s, self.sim_size) for m, s in y])
            y = y.reshape(-1, 1)
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
        results = pd.DataFrame(
            columns=[
                'concentration',
                'mean_pascal',
                'std_pascal'
            ]
        )
        results['concentration'] = X.flatten()
        results['mean_pascal'] = predictions_mean.flatten()
        results['std_pascal'] = predictions_std.flatten()
        if self.save_name is not None:
            results.to_csv(f'results\\tables\\{self.save_name}.csv')
        return results

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
            np.clip(predictions_mean - (predictions_std * 2.5), 0, np.inf),
            alpha=0.25
        )
        plt.scatter(X, y, c='r')
        plt.xlabel('Concentration')
        plt.ylabel('Estimated Pascal')
        if self.save_name is not None:
            plt.title(f'Estimated Pascal for {self.save_name}')
            plt.savefig(f'results\\images\\{self.save_name}.png')
        plt.show()
