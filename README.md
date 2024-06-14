# Ultra-marginal Feature Importance (UMFI)

This is a new Python package for computing ultra-marginal feature importance ([paper link](https://proceedings.mlr.press/v206/janssen23a.html)), which measures the importance score that a specified feature has on a response variable. In contrast to feature importance scores that seek to explain the model, ultra-marginal feature importance was developed to explain the data. 

Code for reproducing the simulated data experiments and real data experiments from our original paper ([paper link](https://proceedings.mlr.press/v206/janssen23a.html)) are included in `experiments.py` and can be run via `main.py`. `experiments.py` also includes simulated data experiments from the papers "Information-Theoretic State Variable Selection for Reinforcement Learning" by Westphal et al. ([paper link](https://arxiv.org/abs/2401.11512)) and "Partial Information Decomposition for Data Interpretability and Feature Selection" by Westphal et al. ([paper link](https://arxiv.org/abs/2405.19212)). 

### Implementation of UMFI Scores

The UMFI score of a given feature $x$ with respect to response $y$ is computed by estimating the increase in universal predictive power $\nu$ when we include $x$ in the feature set $F$, versus when we exclude it. However, in contrast to other ablation methods, we make sure to preprocess the feature set $F$ to optimally remove dependencies on $x$ in order to isolate its impact on predicting the response given the other variables. We denote this preprocessing $S^F_{x}$. The implementation of UMFI is done in `UMFI.py`, where $\nu$ is approximated via random forest OOB accuracy or extra random trees OOB accuracy.

$$U^{F,y}_\nu (x)= \nu(S^F_x \cup x) - \nu(S^F_x)$$

We preprocess the data using either linear regression or optimal transport (from "An algorithm for removing sensitive information: application to race-independent recidivism prediction" by Johndrow et al.). These implementations are found in `preprocess_LR.py` and `preprocess_OT.py` respectively. In this code package, we have updated the optimal transport preprocessing method to also account for cases where variables are sampled from discrete random variables.

Users can plot the UMFI scores using the function `plot_results` in `utils.py`.

The required packages to run this code are given in `umfi.yml`. The compatible conda environment can be created by running `conda env create -f umfi.yml`
