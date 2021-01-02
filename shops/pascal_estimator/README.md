
# Polynomial Bayesian Ridge Regression for Rigidity Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Small project showing the application of Polynomial Bayesian Ridge Regression for estimating the rigidity of a gel given the concentration of its components.

## Motivation

Let's immagine that we are working with a gel which rigidity in Pascal (Pa) is a function of the concentration of a component X present in it. We do not know how this function looks like and the maifacturer producing the gel only provides the mean (and standard deviation) Pa for a restricted number of concentration values. Unfortunately for our use case we need to know the rigidity of the gel for all the possible concentration values of X.  
  
One possibility for overcoming this inconvenience could be to infer the relationship between Pa and the concentration of components X using the data provided by the manifacturer. This can be done through regression anlaysis satisfying however two major requirements:
  
1. Being able to estimate a non-linear relationship if present.
2. Being applicable in data-scarce settings.

In this view, fitting a [polynomial regression](https://en.wikipedia.org/wiki/Polynomial_regression#Alternative_approaches) within a [bayesian framework](https://en.wikipedia.org/wiki/Bayesian_statistics) seems to be one of the most straightforward solutions.

## Features

1. Create a flexible model for estimating the rigidity of a gel given the concentration of its components.
2. Explicitly represent uncertainity in the estimation.
3. Possibility to simulate synthetic data in case when only descriptive statistics are provided.
4. Possibility to tune the hyper-parameters of the model used for estimation.

## How to use  

```python
CONCENTRATION = [1, 3, 7]
PASCAL = [[5, 0.1], [55, 11], [341, 50]]
```

```python
import numpy as np

from modules.bayesian_estimators import BayesianPolyEst

model = BayesianPolyEst(
    degree=2,
    normalize=True,
    n_iter=1000
)
```

```python
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
  <img width="400" height="400" src="https://github.com/vb690/bazaar/blob/master/shops/pascal_estimator/results/images/chemical_X.png">
</p>   
  
```python
results = model.predict(
    X=np.array([2, 5, 16, 21]).reshape(-1, 1),
    verbose=True
)
```
| Concentration | Mean Pascal | Std Pascal |
|---------------|-------------|------------|
|       2       |      22     |     26     |
|       5       |     165     |     26     |
|       16      |     1856    |     47     |
|       21      |     3229    |     85     |

## Shortcomings

1. The modelling approach is over-simplicistic.  
2. Polynomial regressions can behave in an [un-predictable manners](https://en.wikipedia.org/wiki/Polynomial_regression#Alternative_approaches) (expecially at high degrees) .
3. There is no control on the type of priors used by the model (defaults provided by scikit-learn).

## Credits

## License

[The MIT License](https://github.com/vb690/bazaar/blob/master/LICENSE)
