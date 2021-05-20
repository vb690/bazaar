import numpy as np

import matplotlib.pyplot as plt

from modules.optimizers import maximum_likelyhood, metropolis_hastings
from modules.utils import plot_solution

y = np.random.normal(10, 1, size=100)

mcmc_mu = metropolis_hastings(
    y=y,
    samples=1000
)
fig = plot_solution(
    mu=10,
    variance=1,
    approx_solution=mcmc_mu
)
plt.show()
mle_mu = maximum_likelyhood(
    y=y,
    boot=1000
)
fig = plot_solution(
    mu=10,
    variance=1,
    approx_solution=mle_mu
)
plt.show()
