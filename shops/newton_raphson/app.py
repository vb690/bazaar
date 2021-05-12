import numpy as np

import matplotlib.pyplot as plt

from modules.functions import normal_log_likelyhood
from modules.optimizers import newton_method
from modules.utils import plot_solution


solution = newton_method(
    normal_log_likelyhood,
    x_candidate=10,
    maxiter=10000,
    return_history=False,
    y=np.random.normal(10, 1, size=100)
)
fig = plot_solution(
    normal_log_likelyhood,
    10,
    solution,
    y=np.random.normal(10, 1, size=100)
)

plt.show()
