from time import time

import pandas as pd

import pymc3 as pm


def get_data(path="data//breast_cancer.csv"):
    """Get data for the logistic model
    
    Args:
        path (string): location of the dataset
        
    Returns:
        X (array): array of standardized exogenous variables
        y (array): array of endogenous variables
    """
    df = pd.read_csv('data//breast_cancer.csv')
    X = df.drop('diagnosis', axis=1).values
    X = X - X.mean() / X.std()
    y = df['diagnosis'].values
    return X, y

def logistic_regression(X, y):
    """Create logistic model

    Args:
        X (array): exogenous variables
        y (array): endogenous variables

    Returns:
        logistic (object): Created PyMC3 model
    """
    with pm.Model() as logistic:
        intercept = pm.Normal(
            'Intercept',
            mu=0,
            sigma=5
        )
        slopes = pm.Normal(
            'Slopes',
            mu=0,
            sigma=5,
            shape=(X.shape[1])
        )

        p = pm.math.sigmoid(intercept + pm.math.dot(X, slopes))

        observed = pm.Bernoulli('observed', p=p, observed=y)

    return logistic


def profiler(X, y, max_iters=10):
    """Profile sampling from max_iters logistic models

    Args:
        X (array): exogenous variables
        y (array): endogenous variables
        max_iters: number of times we will run the sampler
            for logistic regression
    Returns:
        times (list): execution times for each iteration
        traces (multitrace): traces generated but the last sampling
    """
    times = []
    for iteration in range(max_iters):

        start = time()
        with logistic_regression(X, y):
            traces = pm.sample(target_accept=.90, return_inferencedata=False)
        end = time()
        times.append(end - start)

    return times


def main():
    X, y = get_data()
    times = profiler(X, y, max_iters=10)
    print(times)

    
##############################################################################


if __name__ == '__main__':
    main()