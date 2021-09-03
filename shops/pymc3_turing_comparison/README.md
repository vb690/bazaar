# PyMC3 vs Turing.jl <br /> The fastest sampler alive.
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Motivation

This small project aims to compare the speed of the [No U-Turn-Sampler (NUTS)](https://arxiv.org/abs/1111.4246) in [PyMC3](https://docs.pymc.io/) and [Turing.jl](https://turing.ml/dev/) when fitting a logistic regression to the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

# Experiment Settings

### Sampling
**MCMC Chains** 4  
**Sampler** NUTS  
**Warmup Samples** 1000  
**Effective Samples** 1000  
**Target Acceptance** 0.90

### System
**OS** Windows 10 Home Version 10.0.19041  
**Processor** Intel Core i7 CPU 2.8 GHz 4 Cores 8 Logical Proccessors 

### PyMC3
**Python Version** 3.8.8  
**PyMC3 Version**  3.11.2 installed through **conda** 4.9.2  
**Number of Cores** 4

### Julia
**Julia Version** 1.6  
**Turing.jl Version** 0.15.17  
**Number of Processes (using MCMCDistributed)** 4  

# Results

<p align="center">   
  <img width="400" height="400" src="https://github.com/vb690/bazaar/blob/master/shops/pymc3_turing_comparison/results/figures/boxplot_comp.png">
</p>   

We also compared the two frameworks using the approach presented in [this tuorial on the PyMC3 documentation](https://docs.pymc.io/notebooks/BEST.html)


<p align="center">   
  <img width="900" height="400" src="https://github.com/vb690/bazaar/blob/master/shops/pymc3_turing_comparison/results/figures/posteriors.png">
</p>   
