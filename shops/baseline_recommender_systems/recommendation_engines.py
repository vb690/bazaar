"""Module implementing recommendation engines.

This module contains a collection of classes implementing
recommendation engines. When queried each recommendation
engine will provide a broad area of recommendation and a
specific product inside that area.

  Example:

  recommender = MyRecommender()
  area, product = recommender.recommend()
"""

import numpy as np

from scipy.special import softmax


class SoftmaxRecommender:
    """Class implementing a recommendation engine based on
    temperated softmax.

    The recommendation engine will first individuate the most
    likely area of interest associated to a specific query
    based on the frequency by which each area has been
    previously visited.

    Subsequently, recommendation engine will individuate
    the most likely product associated to the chosen area
    based on the frequency by which each product has been
    acquired.

    The change from frequencies to probabilities is carried
    out using a softmax function which is subsequently
    temperated for increasing variability in the recommendations.

    Attributes:
        areas_dict: a dictiory, keys are query ids values are
                iterables reporting the frequencies associated
                to each area.

        products_dict: a dictiory, keys are areas values are
                    iterables reporting the frequencies for
                    each product in that specific area.

        areas_mapper: a dictionary, keys are indices associated
            to each area values are names for each area.

        products_mapper:a dictionary, keys are indices associated
            to each product values are names for each product.
    """
    def __init__(self, areas_dict, products_dict,
                 areas_mapper, products_mapper):
        """Inits SoftmaxRecommender.
        """
        self.areas_dict = areas_dict
        self.products_dict = products_dict

        self.areas_mapper = areas_mapper
        self.products_mapper = products_mapper

    @staticmethod
    def temperated_prediction(probabilities, temperature=1.0):
        """Static method for sampling from the temperated output
        of a softmax function.

        Higher temperature = more variability, more randomness
        Lower temperature  = less variability, more repetition

        Args:
            - probabilities: array-like, of probability
            - temperature: float, controlling the probability
        Returns:
            - The argmax of the temperated softmax output
        """
        probabilities = np.asarray(probabilities).astype('float64')
        # take the log of the probabilities
        probabilities = np.log(probabilities) / temperature
        # take the exponentiated result
        exp_probabilities = np.exp(probabilities)
        p_probabilities = exp_probabilities / np.sum(exp_probabilities)
        predictions = np.random.multinomial(1, p_probabilities, 1)
        return np.argmax(predictions)

    def recommend(self, query_id, temperature_area=1,
                  temperature_product=1):
        """Perform the recommendation.
        Args:
            - query_id: integer or string, key associated to a specific query
            - temperature_area: float, temperature for computing the
                area softmax
            - temperature_product: float, temperature for computing the
                product softmax

        Returns
            - rec_area: string, recommended area
            - rec_product: string, recommended product
        """
        # transform frquencies to probabilities for a specific query
        area_probabilities = softmax(self.areas_dict[query_id])
        # find the most probable area applying temperature for variability
        area_recommendation = self.temperated_prediction(
            probabilities=area_probabilities,
            temperature=temperature_area
        )

        # transform frquencies to probabilities for a specific area
        product_probabilities = softmax(
            self.products_dict[area_recommendation]
        )
        # find the most probable product applying temperature for variability
        product_recommendation = self.temperated_prediction(
            probabilities=product_probabilities,
            temperature=temperature_product
        )

        # map indices to strings
        rec_area = self.areas_mapper[area_recommendation]
        rec_product = self.products_mapper[rec_area][product_recommendation]

        return rec_area, rec_product
