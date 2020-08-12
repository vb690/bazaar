import numpy as np

from modules.estimators import BayesianPolyEst

CONC = np.array(
    [1] * 100 +
    [3] * 100 +
    [7] * 100
).reshape(-1, 1)

PASC = np.array(
    [
        np.random.normal(5, 0.1, 100),
        np.random.normal(55, 11, 100),
        np.random.normal(341, 50, 100)
    ]
).reshape(-1, 1)

model = BayesianPolyEst(
    degree=2,
    normalize=True,
    n_iter=1000
)
model.fit(
    X=CONC,
    y=PASC
)
model.validate(
    X=np.array([1, 3, 7]),
    y=np.array([5, 55, 341])
)

model.predict(
    X=np.array([2]).reshape(-1, 1),
    verbose=True
)
