import numpy as np

from utils import compute_nll


def linear_reg(X, y, coefs):
    """
    """
    intercept = coefs[0]
    slopes = coefs[1:]

    y_hat = intercept + np.dot(slopes, X)

    nll = compute_nll(
        y_hat=y_hat,
        y=y
    )

    return nll
