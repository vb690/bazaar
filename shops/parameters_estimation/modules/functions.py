import numpy as np

from scipy.stats import norm


def normal_log_likelyhood(mu, y, sigma=1):
    """Function returning the log-likelyhood of y given a adtribution with mu
    and sigma parameters.

    Args:
        - mu: a float, specifying the mu paramter for the normal distribution.
        - y: a numpy array, specifying the samples for which we want to know
        the log likelyhood.
        - sigma: a float, specifying the sigma paramter for the normal
        distribution.
    """
    frozen_norm = norm(mu, sigma)
    ll = frozen_norm.logpdf(y)
    return np.sum(ll)
