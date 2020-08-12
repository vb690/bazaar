import numpy as np

from modules.estimators import BayesianPolyEst

CONCENTRATION = [1, 3, 7]
PASCAL = [[5, 0.1], [55, 11], [341, 50]]

model = BayesianPolyEst(
    degree=2,
    normalize=True,
    n_iter=1000
)
model.fit(
    X=CONCENTRATION,
    y=PASCAL,
    simulate=True,
    save_name='chemical_X'
)
model.validate(
    X=np.array([1, 3, 7]),
    y=np.array([5, 55, 341])
)

results = model.predict(
    X=np.array([2, 5, 16, 21]).reshape(-1, 1),
    verbose=True
)
