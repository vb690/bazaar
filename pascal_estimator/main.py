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
).flatten()

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
    X=np.array([i for i in range(0, 10)]).reshape(-1, 1),
    y=PASC.mean(axis=0)
)

model.predict(
    X=np.array([2]).reshape(-1, 1),
    verbose=True
)
