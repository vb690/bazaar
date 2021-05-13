from tqdm import tqdm

import numpy as np
from scipy.stats import norm

from .utils import central_difference


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


def mcmc(y, samples=1000, mu_init=0, warm_up=1000,
         proposal_width=5, mu_prior_mu=0, mu_prior_sd=10):
    """
    """
    mu_current = mu_init
    posterior = [mu_current]
    for sample in tqdm(range(warm_up + samples)):

        # suggest new position
        mu_proposal = norm(mu_current, proposal_width).rvs()

        # Compute likelihoods
        likelihood_current = norm(mu_current, 1).logpdf(y).sum()
        likelihood_proposal = norm(mu_proposal, 1).logpdf(y).sum()

        # Compute prior probability of current and proposed mu
        prior_current = norm(mu_prior_mu, mu_prior_sd).pdf(mu_current)
        prior_proposal = norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)

        p_current = likelihood_current * prior_current
        p_proposal = likelihood_proposal * prior_proposal

        # Accept proposal?
        p_accept = p_proposal / p_current

        # Usually would include prior probability,
        # which we neglect here for simplicity
        accept = np.random.rand() < p_accept

        if accept:
            # Update position
            mu_current = mu_proposal

        if sample > warm_up:
            posterior.append(mu_current)

    return np.array(posterior)
