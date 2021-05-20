from tqdm import tqdm

import math as m

import numpy as np
from scipy.stats import norm

from .utils import central_difference
from .log_likelyhoods import normal_log_likelyhood


def maximum_likelyhood(y, mu_init=0., tol=1e-4, maxiter=100, verbose=0,
                       boot=1000):
    """Maximum likelyhood estimation of mu using newton rhapson. This is going
    to be ugly as we will bootstrap for obtining the Confidence interval

    Args:
        y: a numpy array of the observed samples originated by the distribution
            we are trying to approximate.
        mu_init: a float, initial guess for the parameter mu.
        tol: a float controlling the minimum improvement for determining
             the convergence of the algorithm
        maxiter: an int controlling the maximum number of iterations before the
                 the algorithm performs before stopping
        verbose: an int controlling the ammount of information logged
                 during iteration
        boot: an int, number of time we will compute the maximum_likelyhood
        from a bootstrap sample


    Returns:
        bootstrapped_mus: a list or float reporting the bootstrapped value of
        mu.
    """
    bootstrapped_mus = []
    for n in tqdm(range(boot)):

        mu_current = mu_init
        bootstrap_sample = np.random.choice(y, len(y))

        for iteration in range(int(maxiter)):

            derivative = central_difference(
                f=normal_log_likelyhood,
                x=mu_current,
                y=bootstrap_sample
            )
            if m.isclose(derivative, 0):
                print('Warning, derivative is zero')
                return mu_current

            mu_new = mu_current - \
                normal_log_likelyhood(mu=mu_current, y=bootstrap_sample) \
                / derivative
            tol_new = abs(mu_new - mu_current)
            mu_current = mu_new

            if verbose > 0:
                print(f'Tol {tol_new} - Solution {mu_current}')
            if tol_new < tol:
                break

        bootstrapped_mus.append(mu_current)

    return np.array(bootstrapped_mus)


def metropolis_hastings(y, mu_init=0., warm_up=1000, samples=1000,
                        proposal_width=0.1, prior_mu=0., prior_sigma=10.):
    """Markov Chain Monte Carlo method for approximating the posterior
    distribution of the parameters defining the distribution that generated y.
    In reality we are simplifying the task here assuming sigma being fixed and
    trying to approximate the posterior for mu.

    Args:
        y: a numpy array of the observed samples originated by the distribution
            we are trying to approximate.
        mu_init: a float, initial guess for the parameter mu.
        warm_up: an int, number of iterations for warming up the algorithm.
        samples: an int: number of samples taken from the posterior.
        proposal_width: a float, the standard deviation of the
        proposal distribution from which values of mu are sampled.
        prior_mu: a float, the mu parameter of the normal prior of mu.
        prior_sigma: a float, the sigma parameter of the normal prior of mu.

    Returns:
        trace: a numpy array, samples from the posterior of mu.
    """
    # we keep everything on the log scale
    mu_current = mu_init
    trace = [mu_current]
    for sample in tqdm(range(warm_up + samples)):

        # suggest new position
        mu_proposal = norm(mu_current, proposal_width).rvs()

        # Compute likelihoods
        ll_current = normal_log_likelyhood(
            mu=mu_current,
            sigma=1,
            y=y
        )
        ll_proposal = normal_log_likelyhood(
            mu=mu_proposal,
            sigma=1,
            y=y
        )

        # Compute prior probability of current and proposed mu
        prior_current = norm(prior_mu, prior_sigma).logpdf(mu_current)
        prior_proposal = norm(prior_mu, prior_sigma).logpdf(mu_proposal)

        p_current = ll_current + prior_current
        p_proposal = ll_proposal + prior_proposal

        if p_proposal > p_current:
            mu_current = mu_proposal
        else:
            if np.random.rand() < np.exp(p_proposal - p_current):
                mu_current = mu_proposal

        if sample > warm_up:
            trace.append(mu_current)

    return np.array(trace[1:])
