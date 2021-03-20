import numpy as np

import pymc3 as pm

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


@st.cache
def compute_r2(y, y_hat):
    """
    """
    ss_res = np.sum(y - y_hat) ** 2
    ss_tot = np.sum(y - np.mean(y)) ** 2
    r2 = ss_res / ss_tot
    return round(r2, 2)


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
    """
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

    trace = {
        'Intercept': trace['Intercept'],
        'Slope': trace['Slope']
    }

    return trace


def get_anscombe_quartet_traces(quartet, mu_intercept=0, sd_intercept=1,
                                mu_slope=0, sd_slope=1, mu_error=1, **kwargs):
    """
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


def plot_anscombe_quartet(quartet, quartet_traces, **kwargs):
    """Visualize the anscombe quartet

    Args:
        - quartet: a disctionary, anscombe quartet produced by
                   get_anscombe_quartet()
        - kwargs: keyward arguments to be passed to plt.subplots()

    Returns:
        - fig: a matplolib figure obejct containing the anscombe plot.
    """
    quartet_components = ['I', 'II', 'III', 'IV']
    fig, axs = plt.subplots(**kwargs)
    for component, ax in zip(quartet_components, axs.flatten()):

        component_intercept = quartet_traces[component]['Intercept']
        component_slope = quartet_traces[component]['Slope']

        component_x = quartet[component]['x']
        component_y = quartet[component]['y']
        component_predictor = np.linspace(
            2,
            20,
            1000
        )
        component_line = component_intercept.reshape(-1, 1) +  \
            component_slope.reshape(-1, 1) * component_predictor.reshape(1, -1)
        lower, upper = np.percentile(component_line, [5, 90], axis=0)
        component_line = component_line.mean(axis=0)

        ax.plot(
            component_predictor,
            component_line,
            c='r',
            linewidth=0.5
        )
        ax.fill_between(
            component_predictor,
            lower,
            upper,
            color='r',
            alpha=0.1
        )
        ax.scatter(
            component_x,
            component_y,
            facecolors='none',
            edgecolors='k',
            s=20,
            zorder=1
        )

        ax.set_xlim(2, 20)
        ax.set_ylim(2, 14)
        ax.set_title(component)

    fig.text(0.5, 0.00, 'X', ha='center')
    fig.text(0.00, 0.5, 'Y', va='center')
    plt.tight_layout()

    return fig

###############################################################################


if __name__ == '__main__':

    st.title("Bayesian Anscombe's Quartet")

    quartet = get_anscombe_quartet()

    col1, col2 = st.beta_columns(2)
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
        '\u03B2 Error',
        min_value=0.1,
        max_value=10.,
        value=1.,
        step=0.1
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

    fig = plot_anscombe_quartet(
        quartet=quartet,
        quartet_traces=quartet_traces,
        nrows=2,
        ncols=2,
        figsize=(7, 7),
        sharex=True,
        sharey=True
    )

    col1.pyplot(fig)
