import numpy as np

import matplotlib.pyplot as plt

from modules.functions import normal_log_likelyhood
from modules.optimizers import newton_raphson, mcmc
from modules.utils import plot_solution

y = np.random.normal(10, 1, size=100)

plt.plot(mcmc(
    y=y,
    samples=1000
))
plt.show()

solution = newton_raphson(
    normal_log_likelyhood,
    x_candidate=1,
    maxiter=1000,
    return_history=False,
    y=y
)
fig = plot_solution(
    f=normal_log_likelyhood,
    solution=10,
    approx_solution=solution,
    y=y
)

plt.show()
