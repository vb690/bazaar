
import matplotlib.pyplot as plt


def plot_solution(f, x, solution):
    """Plot the solution reached for f given the x starting point

    Args:
        f: a function taking x as input
        x: an int or a float initial guess for the solution
        solution: an int or a float solution reached by the algorithm

    Returns:
        fig: a matplotlib figure
    """
    range_x = [i for i in range(-x, x)]
    f_x = [f(i) for i in range_x]
    f_solution = f(solution)

    fig = plt.figure(figsize=(5, 5))
    plt.plot(
        range_x,
        f_x,
    )
    plt.scatter(
        solution,
        f_solution,
        c='r',
        label=f'Reached minimum at {solution}'
    )
    plt.xlabel('$X$')
    plt.ylabel('$f(X)$')
    plt.legend()
    return fig


def central_difference(f, x, h=1e-10):
    """Central difference method for approximating the derivartive of f

    Args:
        f: a function taking x as input
        x: an int or float input to the function
        h: a float small constant used for finding the derivative

    Returns:
        f_prime: approximation of the derivative of f
    """
    f_prime = (f(x+h) - f(x)) / h
    return f_prime
