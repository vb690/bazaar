"""Module containing estimators abstract classes.

This module contains the abstract classes inherithed by the estimators.
Each abstract class provides methods shared across all estimators while also
endorcing the implementation of core methods.

  Typical usage example:

  class ClassEstimator(AbstractClassEstimator):
      ...

"""
from abc import ABC

import numpy as np

import matplolib.pyplot as plt
import seaborn as sns


class AbstractBayesianEstimator(ABC):
    """
    """
    def __init__(self):
        """
        """
        pass

    @staticmethod
    def loo_hyperp_tuning(regressor, X, y, distributions, **kwargs):
        """
        """
        return None

    @staticmethod
    def prediction_printer(X, mean, std):
        for x, m, s in zip(X, mean, std):

            print(f'Estimated Pascal for concentration {x}: {m} + / - {s}')

    def validation_plot(self, X, y, line, predictions_mean, predictions_std):
        """
        """
        sns.set(
            font_scale=2
        )
        plt.figure(figsize=(10, 10))
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

    def fit(self, X, y):
        """
        """
        raise NotImplementedError('Method fit needs to be implemented.')

    def predict(self, X, y):
        """
        """
        raise NotImplementedError('Method predict needs to be implemented.')

    def validate(self, X, y):
        """
        """
        raise NotImplementedError('Method validate needs to be implemented.')
