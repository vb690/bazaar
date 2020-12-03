"""Module containing bayesian estimator classes based on scikit-learn.

This module contains a collection of bayesian estimator classes. Each estimator
is tasked exclusively to perform prediction and no method for parameters
inspection is explicitly implemented. However access to the underlying
scikit-learn estimator is granted throught the get_estimator method.

  Typical usage example:

  estimator = ClassEstimator()

  estimator.fit()
  estimator.validate()
  estimator.predict()
"""

import numpy as np

import pandas as pd

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

from _abstract_estimators import AbstractBayesianEstimator


class BayesianPolyEst(AbstractBayesianEstimator):
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
        for x, m, s in zip(X, mean, std):

            print(f'Estimated Pascal for concentration {x}: {m} + / - {s}')

    def fit(self, X, y, simulate=False, save_name=None, **kwargs):
        """
        """
        setattr(self, 'save_name', save_name)
        if simulate:
            X = np.array([[x] * self.sim_size for x in X])
            X = X.reshape(-1, 1)
            y = np.array([np.random.normal(m, s, self.sim_size) for m, s in y])
            y = y.reshape(-1, 1)
        X_poly = self.transformer.fit_transform(X)
        if 'distributions' in kwargs:
            self.loo_hyperp_tuning(
                regressor=self.regressor,
                X=X_poly,
                y=y,
                **kwargs
            )
        self.regressor.fit(
            X_poly,
            y
        )
        return None

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

        predictions_mean, predictions_std = self.regressor.predict(
            X_poly,
            return_std=True
        )

        self.validation_plot(
            X=X,
            y=y,
            line=line,
            predictions_mean=predictions_mean,
            predictions_std=predictions_std
        )
