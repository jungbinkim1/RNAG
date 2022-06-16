# RNAG
This repository includes an implementation of **Riemannian Nesterov Accelerated Gradient Method (RNAG)** in [(Kim & Yang, 2022)][kim2022]. Our code is based on [Orvieto's code][orvietocode] [(Alimisis et al., 2021)][alimisis2021].

## Requirements
pymanopt==0.2.5

geomstats==2.2.2

torch==1.9.0

scipy==1.5.0

numpy==1.19.5

matplotlib==3.2.2

## Contents of this folder
This folder contains 5 files.

optimizers.py: implementation of various Riemannian optimization algorithms

rayleigh_quotient.py, Karcher_mean_spd.py, Karcher_mean_hyperbolic.py: implementations for numerical experiments in [(Kim & Yang, 2022, Section 7)][kim2022]

limiting_convex.py, limiting_strongly_convex.py: implementations for numerical experiments in [(Kim & Yang, 2022, Appendix G)][kim2022]

utils.py: additional functions

[alimisis2021]: http://proceedings.mlr.press/v130/alimisis21a/alimisis21a-supp.pdf
[kim2022]: https://arxiv.org/pdf/2202.02036.pdf
[orvietocode]: https://github.com/aorvieto/RNAGsDR
