from umfi.utils import is_discrete
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def preprocess_ot(dat, protect, n_quan=None, min_sd=1e-6, discretization_quantiles = 5):
    '''
    This function preprocess the feature set (dat) to remove dependencies between the protected feature (Z)
    and the rest of the features (X) via the pairwise optimal transport F_X|Z

    based on Algorithm 1 from "An algorithm for removing sensitive information: application
    to race-independent recidivism prediction" by James E. Johndrow  and Kristian Lum

    :param dat: A DataFrame or array-like of continuous data.
    :param protect: The column name of the protected attribute.
    :param n_quan: The number of quantiles to use to estimate the conditional CDF (default is ceil(n_samples/150)).
    :param min_sd: The minimum standard deviation that data points within a quantile can have before noise is added (default is 1e-6)
    :param discretization_quantiles: The number of quantiles used to discretize a continuous r.v. into if applicable (default is 5)
    :return: A DataFrame or array-like of the preprocessed dataset with dependencies on protected feature removed
    The Dataframe contains the protected feature as its own column
    '''
    # initialize number of quantiles
    if n_quan is None:
        n_samples = dat.shape[0]
        n_quan = int(np.ceil(n_samples / 150))
    modified_dat = dat.copy()  # Create a copy of the input data
    # we will remove dependencies of the protected feature from all other features
    tomodify = [col for col in dat.columns if col != protect]
    z = dat[protect].values  # Extract the protected attribute as a NumPy array
    quans = np.quantile(z, np.linspace(0, 1, n_quan))  # Calculate quantiles of the protected attribute
    # perform pairwise optimal transport to remove dependencies on z from the other features
    for j in tomodify:
        x = dat[j]
        if is_discrete(x):
            # if x is discrete and z is cts, then we discretize z to approximate the continuous cdf
            if not is_discrete(z):
                ranks = z.argsort().argsort()
                z = np.round(ranks * discretization_quantiles / len(z),0).astype(int)
            newx = discrete_univariate_optimal_transport(x, z)
        else:
            # if x is cts and z is discrete, then we discretize x to approximate the continuous cdf
            if is_discrete(z):
                ranks = x.argsort().argsort()
                x = np.round(ranks * discretization_quantiles / len(z), 0).astype(int)
                newx = discrete_univariate_optimal_transport(x, z)
            else: # use classic quantile estimate if both variables are continuous
                x = dat[j].values
                newx = x.copy().astype(float)  # Create a copy of the feature
                for quan in range(1, n_quan):
                    cur_obs = (z <= min(quans[quan], max(z))) & (z > quans[quan - 1])
                    x_curquan = x[cur_obs]
                    z_curquan = z[cur_obs]
                    if len(x_curquan) == 0 or len(z_curquan) == 0:
                        continue
                    # Add noise if the standard deviation is too low
                    if x_curquan.std() < min_sd:
                        x_curquan = x_curquan.astype(float)
                        x_curquan += np.random.normal(scale=x.std() / len(x), size=len(x_curquan))

                    if z_curquan.std() < min_sd:
                        z_curquan = z_curquan.astype(float)
                        z_curquan += np.random.normal(scale=z.std() / len(z), size=len(z_curquan))

                    # Fit a linear regression model to approximate the conditional distribution
                    model = LinearRegression().fit(z_curquan.reshape(-1, 1), x_curquan)

                    rv = x_curquan - model.predict(z_curquan.reshape(-1, 1)) # residuals
                    newx[cur_obs] = (rv.argsort().argsort() + 1)  # replace x with the ranked residuals
        modified_dat[j] = newx
    return modified_dat


def discrete_univariate_optimal_transport(x, z):
    """
    Solves the optimal transport preprocessing problem when both x and z are discrete

    based on Algorithm 1 from "An algorithm for removing sensitive information: application
    to race-independent recidivism prediction" by James E. Johndrow  and Kristian Lum

    :param x: Feature to be transformed.
    :param z: Protected attribute.
    :return: Transformed feature.
    """
    newx = x.astype(float)
    # extract empirical joint observations
    c_pmf = pd.crosstab(x, z)
    c_cmf = c_pmf.cumsum(axis=0).div(c_pmf.sum(axis=0), axis=1)
    for i in range(len(x)):
        xi_min = x[x < x[i]]
        if len(xi_min) > 0:
            xi_min = xi_min.max()
        else:
            xi_min = -np.inf
        l_xi = c_cmf.loc[xi_min, z[i]] if xi_min in c_cmf.index else 0
        r_xi = c_cmf.loc[x[i], z[i]] if x[i] in c_cmf.index else 0
        u_i = np.random.uniform(l_xi, r_xi)
        newx.iloc[i] = u_i
    return newx
