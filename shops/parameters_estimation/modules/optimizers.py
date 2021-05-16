from tqdm import tqdm

import numpy as np
from scipy.stats import norm

from .utils import central_difference
from .functions import normal_log_likelyhood


def newton_method(f, x_candidate, tol=1e-10, maxiter=10, verbose=0,
                  return_history=True, **kwargs):
    """Iterative algorithm for finding the root of f

    Args:
        f: a function taking x_candidate as input
        x_candidate: an int or float, intial guess for the solution
        tol: a float controlling the minimum improvement for determining
             the convergence of the algorithm
        maxiter: an int controlling the maximum number of iterations before the
                 the algorithm performs before stopping
        verbose: an int controlling the ammount of information logged
                 during iteration
        return_history: a bolean specifying if the list of possible solutions
                        is returned

    Returns:
        x: an int or float reporting the solution found by the algorithm
        solutions: a list of int or floats containing the sequence of solutions
                   found during the iteration
    """
    x = x_candidate
    solutions = []
    for iteration in tqdm(range(int(maxiter))):

        x_new = x - f(x, **kwargs) / central_difference(f, x, **kwargs)
        solutions.append(x_new)
        tol_new = abs(x_new - x)
        x = x_new
        if verbose > 0:
            print(f'Tol {tol_new} - Solution {x_new}')
        if tol_new < tol:
            break

    if return_history:
        return x, solutions
    else:
        return x


def mcmc(y, mu_init=0, warm_up=1000, samples=1000,
         proposal_width=0.1, prior_mu=0, prior_sigma=10):
    """
    """
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
