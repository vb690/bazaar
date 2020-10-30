
# Polynomial Bayesian Ridge Regression for Rigidity Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Small project showing the application of Polynomial Bayesian Ridge Regression for estimating the rigidity of a gel given the concentration of their components.

## Motivation

## Features

## Examples

## Installation

## How to use  
  
```python
import numpy as np

from modules.bayesian_estimators import BayesianPolyEst

CONCENTRATION = [1, 3, 7]
PASCAL = [[5, 0.1], [55, 11], [341, 50]]
```

```python
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
```

```python
model.validate(
    X=np.array([1, 3, 7]),
    y=np.array([5, 55, 341])
)
```  
  
<p align="center">   
  <img width="400" height="400" src="https://github.com/vb690/bazaar/blob/master/pascal_estimator/results/images/chemical_X.png">
</p>   
  
```python
results = model.predict(
    X=np.array([2, 5, 16, 21]).reshape(-1, 1),
    verbose=True
)
```

## Credits

## License

[The MIT License](https://github.com/vb690/bazaar/blob/master/LICENSE)
