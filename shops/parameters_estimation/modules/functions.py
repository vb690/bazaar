import numpy as np

from scipy.stats import norm


def normal_log_likelyhood(mu, sigma=1, y=np.random.normal(size=1000)):
    """
    """
    frozen_norm = norm(mu, sigma)
    ll = frozen_norm.logpdf(y)
    return np.sum(ll)
