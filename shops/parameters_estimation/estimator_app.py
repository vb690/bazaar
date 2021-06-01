import numpy as np

import streamlit as st

from modules.optimizers import maximum_likelyhood, metropolis_hastings
from modules.utils import plot_solution


def run_app():
    """Function running the streamlit app
    """
    st.set_page_config(layout="wide")
    st.title("Parameter Estimation")

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
        value=10,
        step=1
    )

    y = np.random.normal(mu, 1, size=n)

    st.sidebar.title('Metropolis Hastings')
    mu_init = st.sidebar.slider(
        '\u03BC Init',
        min_value=-100,
        max_value=100,
        value=0,
        step=1
    )
    samples = st.sidebar.number_input(
        'Samples',
        min_value=100,
        max_value=10000,
        value=100
    )
    warm_up = st.sidebar.number_input(
        'Warm-Up',
        min_value=100,
        max_value=10000,
        value=100
    )
    proposal_width = st.sidebar.slider(
        'Proposal Width',
        min_value=0.,
        max_value=10.,
        value=0.1,
        step=0.1
    )
    mu_prior = st.sidebar.slider(
        '\u03BC Prior',
        min_value=-100,
        max_value=100,
        value=0,
        step=1
    )
    sd_prior = st.sidebar.slider(
        '\u03C3 Prior',
        min_value=0,
        max_value=100,
        value=10,
        step=1
    )

    run_sampler = st.sidebar.button(
        'Run Sampler'
    )
    if run_sampler:
        mcmc_mu = metropolis_hastings(
            y=y,
            mu_init=mu_init,
            warm_up=warm_up,
            samples=samples,
            proposal_width=proposal_width,
            prior_mu=mu_prior,
            prior_sigma=sd_prior
        )
        fig = plot_solution(
            mu=mu,
            variance=1,
            approx_solution=mcmc_mu
        )

        st.pyplot(fig)

    # mle_mu = maximum_likelyhood(
    #     y=y,
    #     boot=100
    # )
    # fig = plot_solution(
    #     mu=mu,
    #     variance=1,
    #     approx_solution=mle_mu
    # )


if __name__ == '__main__':
    run_app()
