# Maximum Likelyhood vs Bayesian Parameter Estimation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Motivation
This small projects aims to compare the results provided by Maximum Likelyhood and Metropolis Hastings in estimating the parameter `mu` of a normal distribution with unit scale.

## Features

1. Maximum likelyhood estimation algorithm based on the Newton-Raphson method. 
   * Uncertainty estimated through (very inefficient) bootstrapping.
3. Posterior estimation algorithm based on the metropolis-hastings method.
4. Interactive app for estimating and visualizing the parameter `mu` using both algorithms.

## How to Use

1. Download your local version of the repository
2. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)
3. Open the Anaconda Powershell Prompt in the repository directory
```sh
# create anaconda environment
conda create -n app_env python=3.8

# activate the environment
conda activate app_env

# install the requirements
conda install -c conda-forge --file requirements.txt
```
4. Run the app through
```sh
streamlit run estimator_app.py
```
