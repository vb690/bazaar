
import matplotlib.pyplot as plt
import seaborn as sns


def plot_solution(mu, variance, approx_solution, **kwargs):
    """Plot the solution reached for f given the x starting point
    """
    fig = plt.figure(figsize=(5, 5))
    sns.kdeplot(
        approx_solution
    )
    plt.axvline(
        mu,
        c='r',
        label=f'Original mu {mu}'
    )
    plt.legend()
    return fig


def central_difference(f, x, h=1e-10, **kwargs):
    """Central difference method for approximating the derivartive of f

    Args:
        f: a function taking x as input
        x: an int or float input to the function
        h: a float small constant used for finding the derivative

    Returns:
        f_prime: approximation of the derivative of f
    """
    f_prime = (f(x+h, **kwargs) - f(x, **kwargs)) / h
    return f_prime
