from tqdm import tqdm

from utils import central_difference


def newton_method(f, x_candidate, tol=1e-10, maxiter=10, verbose=0,
                  return_history=True):
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

        x_new = x - f(x) / central_difference(f, x)
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
