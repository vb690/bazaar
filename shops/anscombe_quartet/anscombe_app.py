import numpy as np
from scipy.stats.stats import pearsonr

import pandas as pd

import pymc3 as pm

import matplotlib.pyplot as plt
import streamlit as st


@st.cache
def compute_r2(y, y_hat):
    """
    Computing the coefficient of determination give y and y_hat. This is
    the proportion of the variance in the dependent variable that is
    predictable from the independent variable.

        Args:
            - y: is an array-like indicating the values of the dependent
                variable.
            - y_hat: is an array-like indicating the estimated values.

        Returns:
            - r2: is a float indicating the coefficient of determination
    """
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


@st.cache
def compute_anscombe_descriptives(quartet, quartet_traces):
    """Compute a series of descriptive statistics from the 4 anscombe's
    datasets and the linear models associated to them.

        Args:
            - quartet: is a dictionary containing the 4 anscombe' datasets as
                produced by get_anscombe_quartet.
            - quartet_traces: is a dictionary containing PyMC3 multitrace
                obejcts obtainined by fitting a linear model to each of the
                anscombe's datasets.

        Returns:
            - descriptives: is a pandas DataFrame reporting mean, variance,
                correlation and R^2 for each of the anscombe's datasets.
    """
    descriptives = pd.DataFrame(
        columns=[
            'Dataset',
            '\u03BC X',
            '\u03BC Y',
            '\u03C3\u00B2 X',
            '\u03C3\u00B2 Y',
            '\u03C1(X, Y)',
            'R\u00B2'
        ]
    )
    for index, component in enumerate(['I', 'II', 'III', 'IV']):

        component_x = quartet[component]['x']
        component_y = quartet[component]['y']
        component_intercept = quartet_traces[component]['Intercept']
        component_slope = quartet_traces[component]['Slope']
        component_line = component_intercept.reshape(-1, 1)  \
            + component_slope.reshape(-1, 1) * component_x.reshape(1, -1)
        component_line = component_line = component_line.mean(axis=0)

        component_mu_x = round(np.mean(component_x), 3)
        component_mu_y = round(np.mean(component_y), 3)
        component_var_x = round(np.var(component_x, ddof=1), 3)
        component_var_y = round(np.var(component_y, ddof=1), 3)
        component_rho_x_y = round(pearsonr(component_x, component_y)[0], 3)
        component_r2 = round(compute_r2(component_y, component_line), 2)

        descriptives.loc[index] = [
            component,
            component_mu_x,
            component_mu_y,
            component_var_x,
            component_var_y,
            component_rho_x_y,
            component_r2,
        ]

    descriptives = descriptives.transpose()
    descriptives.rename(columns=descriptives.iloc[0], inplace=True)
    descriptives.drop('Dataset', axis=0, inplace=True)
    descriptives = descriptives.transpose()

    return descriptives


@st.cache
def get_anscombe_quartet():
    """Retrieve an anscombe quartet
    https://en.wikipedia.org/wiki/Anscombe%27s_quartet

    Returns:
        - quartet: a disctionary, keys are the names of the datasets values
                   are dictionaries containing the x and y values for the
                   datasets.
    """
    quartet = {
        'I': {
            'x': np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]),
            'y': np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96,
                          7.24, 4.26, 10.84, 4.82, 5.68])
        },
        'II': {
            'x': np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]),
            'y': np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10,
                           6.13, 3.10, 9.13, 7.26, 4.74])
        },
        'III': {
            'x': np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]),
            'y': np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84,
                           6.08, 5.39, 8.15, 6.42, 5.73])
        },
        'IV': {
            'x': np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]),
            'y': np.array([6.58, 5.76, 7.71, 8.84, 8.47,
                           7.04, 5.25, 12.50, 5.56, 7.91, 6.89])
        }
    }
    return quartet


def linear_regression(x, y, mu_intercept, sd_intercept,
                      mu_slope, sd_slope, mu_error,  **kwargs):
    """Fit a linear regression within a Bayesian framework. All the priors are
    assumed to be gaussians excpet for the erro which is HalfCauchy
    distributed.

        Args:
            - x: a numpy array specifying values for the indipendent variable.
            - y: a numpy array specifying values for the dependent variable.
            - mu_intercept: a float specifying the mu for the intercept prior.
            - sd_intercept: a float specifying the sd for the intercept prior.
            - mu_slope: a float specifying the mu for the slope prior.
            - sd_slope: a float specifying the sd for the slope prior.
            - mu_error: a float specifying the mu for the error prior.
            - **kwrgs: keeyward arguments passed to PyMC3 sample function.

        Returns:
            - trace: a PyMC3 multitrace object containing the posteriro for all
                the estimated parameters for the regression model.
    """
    with pm.Model() as linear_model:

        intercept = pm.Normal(
            mu=mu_intercept,
            sd=sd_intercept,
            name='Intercept'
        )

        slope = pm.Normal(
            mu=mu_slope,
            sd=sd_slope,
            name='Slope'
        )

        mu_model = pm.Deterministic(
            var=intercept + slope*x,
            name='Model Mu'
        )

        sd_model = pm.HalfCauchy(
            beta=mu_error,
            name='Model Sigma'
        )

        obs = pm.Normal(
            mu=mu_model,
            sd=sd_model,
            name='Y',
            observed=y
        )

    with linear_model:

        trace = pm.sample(**kwargs)
        pp = pm.sample_posterior_predictive(
            trace,
            var_names=['Intercept', 'Slope', 'Y']
        )

    trace = {
        'Intercept': pp['Intercept'],
        'Slope': pp['Slope'],
        'Y': pp['Y']
    }

    return trace


def get_anscombe_quartet_traces(quartet, mu_intercept=0, sd_intercept=1,
                                mu_slope=0, sd_slope=1, mu_error=1, **kwargs):
    """Fit a bayesian linear regression to each anscombe's dataset and return
    the relative multitrace.

        Args:
            -quartet: is a dictionary containing the 4 anscombe' datasets as
                produced by get_anscombe_quartet.
            - mu_intercept: a float specifying the mu for the intercept prior.
            - sd_intercept: a float specifying the sd for the intercept prior.
            - mu_slope: a float specifying the mu for the slope prior.
            - sd_slope: a float specifying the sd for the slope prior.
            - mu_error: a float specifying the mu for the error prior.
            - **kwrgs: keeyward arguments passed to PyMC3 sample function.

        Returns:
            - quartet_traces: is a dictionary where keys are names of
                anscombe datasets value PyMC3 multitrace objects.
    """
    quartet_traces = {}
    for component, dataset in quartet.items():

        quartet_traces[component] = linear_regression(
            x=dataset['x'],
            y=dataset['y'],
            mu_intercept=0,
            sd_intercept=1,
            mu_slope=0,
            sd_slope=1,
            mu_error=1,
            **kwargs
        )

    return quartet_traces


def plot_anscombe_quartet(quartet, quartet_traces, show_predictions, **kwargs):
    """Visualize the anscombe quartet

    Args:
        - quartet: a disctionary, anscombe quartet produced by
                   get_anscombe_quartet
        - kwargs: keyward arguments to be passed to plt.subplots

    Returns:
        - fig: a matplolib figure obejct containing the anscombe plot.
    """
    if show_predictions:
        alpha = 0.1
    else:
        alpha = 1
    fig, axs = plt.subplots(**kwargs)
    for component, ax in zip(['I', 'II', 'III', 'IV'], axs.flatten()):

        component_intercept = quartet_traces[component]['Intercept']
        component_slope = quartet_traces[component]['Slope']
        component_predictions = quartet_traces[component]['Y']

        component_x = quartet[component]['x']
        component_y = quartet[component]['y']
        component_predictor = np.linspace(
            2,
            20,
            1000
        )
        component_line = component_intercept.reshape(-1, 1) +  \
            component_slope.reshape(-1, 1) * component_predictor.reshape(1, -1)
        lower_line, upper_line = np.percentile(component_line, [5, 90], axis=0)
        component_line = component_line.mean(axis=0)

        ax.plot(
            component_predictor,
            component_line,
            c='r',
            linewidth=0.5
        )
        ax.fill_between(
            component_predictor,
            lower_line,
            upper_line,
            color='r',
            alpha=0.1
        )
        ax.scatter(
            component_x,
            component_y,
            facecolors='none',
            edgecolors='k',
            s=20,
            zorder=1,
            alpha=alpha
        )
        if show_predictions:
            ax.scatter(
                component_x,
                component_predictions.mean(axis=0),
                facecolors='none',
                edgecolors='r',
                s=20,
                zorder=1
            )
            ax.errorbar(
                x=component_x,
                y=component_predictions.mean(axis=0),
                yerr=component_predictions.std(axis=0),
                linewidth=0.5,
                color='r',
                ls='none'
            )

        ax.set_xlim(2, 20)
        ax.set_ylim(0, 16)
        ax.set_title(component)

    fig.text(0.5, 0.00, 'X', ha='center')
    fig.text(0.00, 0.5, 'Y', va='center')
    plt.tight_layout()

    return fig


def run_app():
    """Function running the streamlit app
    """
    st.set_page_config(layout="wide")
    st.title("Bayesian Anscombe's Quartet")

    quartet = get_anscombe_quartet()

    st.sidebar.title('Regression Priors')
    mu_intercept = st.sidebar.slider(
        '\u03BC Intercept',
        min_value=-10.,
        max_value=10.,
        value=0.,
        step=0.1
    )
    sd_intercept = st.sidebar.slider(
        '\u03C3 Intercept',
        min_value=0.,
        max_value=10.,
        value=1.,
        step=0.1
    )
    mu_slope = st.sidebar.slider(
        '\u03BC Slope',
        min_value=-10.,
        max_value=10.,
        value=0.,
        step=0.1
    )
    sd_slope = st.sidebar.slider(
        '\u03C3 Slope',
        min_value=0.,
        max_value=10.,
        value=1.,
        step=0.1
    )
    mu_error = st.sidebar.slider(
        '\u03B5',
        min_value=0.1,
        max_value=10.,
        value=1.,
        step=0.1
    )
    show_predictions = st.sidebar.checkbox(
        'Show Predictions'
    )

    quartet_traces = get_anscombe_quartet_traces(
        quartet=quartet,
        mu_intercept=mu_intercept,
        sd_intercept=sd_intercept,
        mu_slope=mu_slope,
        sd_slope=sd_slope,
        mu_error=mu_error,
        cores=1,
        target_accept=0.9,
        draws=200,
        tune=200
    )

    descriptives = compute_anscombe_descriptives(
        quartet=quartet,
        quartet_traces=quartet_traces
    )

    fig = plot_anscombe_quartet(
        quartet=quartet,
        quartet_traces=quartet_traces,
        nrows=1,
        ncols=4,
        show_predictions=show_predictions,
        figsize=(12, 3),
        sharex=True,
        sharey=True
    )

    st.pyplot(fig)
    st.dataframe(descriptives)


if __name__ == '__main__':
    run_app()
