import numpy as np

import streamlit as st

from modules.optimizers import maximum_likelyhood, metropolis_hastings
from modules.utils import plot_solution


def run_app():
    """Function running the streamlit app
    """
    st.set_page_config(layout="wide")
    st.title("Single Parameter Estimation From Random Sample")

    mle_estimate, mcmc_estimate = st.beta_columns(2)

    st.sidebar.title('Simulated Data')
    mu = st.sidebar.slider(
        '\u03BC',
        min_value=-100,
        max_value=100,
        value=0,
        step=1
    )
    n = st.sidebar.slider(
        'N',
        min_value=10,
        max_value=100,
        value=30,
        step=1
    )

    run_sampler = st.sidebar.button(
        'Run Sampler'
    )
    y = np.random.normal(mu, 1, size=n)

    with st.sidebar.beta_expander('Metropolis Hastings'):
        mu_init_metr = st.slider(
            '\u03BC First ProposalS',
            min_value=-100,
            max_value=100,
            value=0,
            step=1
        )
        samples = st.number_input(
            'Samples',
            min_value=100,
            max_value=10000,
            value=1000
        )
        warm_up = st.number_input(
            'Warm-Up',
            min_value=100,
            max_value=10000,
            value=1000
        )
        proposal_width = st.slider(
            'Proposal Width',
            min_value=0.,
            max_value=10.,
            value=0.1,
            step=0.1
        )
        mu_prior = st.slider(
            '\u03BC Prior',
            min_value=-100,
            max_value=100,
            value=0,
            step=1
        )
        sd_prior = st.slider(
            '\u03C3 Prior',
            min_value=0,
            max_value=100,
            value=10,
            step=1
        )

    with st.sidebar.beta_expander('Newton-Raphson'):
        mu_init_newt = st.slider(
            '\u03BC Init',
            min_value=-100,
            max_value=100,
            value=0,
            step=1
        )
        tol = st.number_input(
            'Tolerance',
            min_value=1e-9,
            max_value=0.1,
            value=1e-9
        )
        maxiter = st.number_input(
            'Maximum Number of Iterations',
            min_value=10,
            max_value=1000,
            value=100
        )
        boot = st.number_input(
            'Number of Bootstrapped Samples',
            min_value=1,
            max_value=100,
            value=30,
            step=1
        )

    if run_sampler:
        mcmc_mu = metropolis_hastings(
            y=y,
            mu_init=mu_init_metr,
            warm_up=warm_up,
            samples=samples,
            proposal_width=proposal_width,
            prior_mu=mu_prior,
            prior_sigma=sd_prior
        )
        mcmc_fig = plot_solution(
            mu=mu,
            variance=1,
            approx_solution=mcmc_mu
        )
        mcmc_estimate.header('Metropolis-Hastings Estimate')
        mcmc_estimate.pyplot(mcmc_fig)

        mle_mu = maximum_likelyhood(
            y=y,
            mu_init=mu_init_newt,
            boot=boot,
            maxiter=maxiter,
            tol=tol
        )
        mle_fig = plot_solution(
            mu=mu,
            variance=1,
            approx_solution=mle_mu
        )
        mle_estimate.header('Newton-Raphson Estimate')
        mle_estimate.pyplot(mle_fig)


if __name__ == '__main__':
    run_app()
