import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ot import emd
from umfi.utils import is_discrete
from scipy.optimize import linear_sum_assignment

def preprocess_ot(dat, protect, n_quan=100, min_sd=1e-6):
    '''
    This function iteratively preprocesses
    Preprocess the data via the optimal transport solution F_X|Z

    This function removes dependencies between the protected attribute (continuous)
    and the rest of the features via optimal transport or linear regression.

    :param dat: A DataFrame or array-like of continuous data.
    :param protect: The column name of the protected attribute.
    :param n_quan: The number of quantiles to use to estimate the conditional CDF (default is ceil(n_samples/150)).
    :param min_sd: The minimum standard deviation that data points within a quantile can have before noise is added (default is 1e-6)
    :return: A DataFrame or array-like of data with the dependencies between the protected attribute and the rest of the features removed.
    '''
    if n_quan is None:
        n_samples = dat.shape[0]
        n_quan = int(np.ceil(n_samples / 150))

    modified_dat = dat.copy()  # Create a copy of the input data
    # we will remove dependencies of the protected feature from all other features
    tomodify = [col for col in dat.columns if col != protect]
    z = dat[protect].values  # Extract the protected attribute as a NumPy array
    quans = np.quantile(z, np.linspace(0, 1, n_quan))  # Calculate quantiles of the protected attribute

    for j in tomodify:
        x = dat[j]
        if is_discrete(x):
            newx = discrete_transform(x, z)
        else:
            x = dat[j].values  # Extract the feature to modify as a NumPy array
            newx = x.copy()  # Create a copy of the feature
            for quan in range(1, n_quan):
                cur_obs = (z <= quans[quan]) & (z > quans[quan - 1])

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
                # Calculate conditional CDF
                rv = x_curquan - model.predict(z_curquan.reshape(-1, 1))
                condF = (rv.argsort().argsort() + 1) / len(rv)
                newx[cur_obs] = condF
        modified_dat[j] = newx

    return modified_dat


def discrete_transform(x, z):
    """
    Applies discrete transformation to the data using cross-tabulation and cumulative marginal frequencies.

    :param x: Feature to be transformed.
    :param z: Protected attribute.
    :return: Transformed feature.
    """
    newx = x.astype(float)
    c_pmf = pd.crosstab(x, z)
    c_cmf = c_pmf.cumsum(axis=0).div(c_pmf.sum(axis=0), axis=1)
    for i in range(len(x)):
        xi_min = x[x < x[i]]
        xi_min = xi_min.max() if len(xi_min) > 0 else -1e12
        left = c_cmf.loc[xi_min, z[i]] if xi_min in c_cmf.index else 0
        right = c_cmf.loc[x[i], z[i]] if x[i] in c_cmf.index else 0
        newx.iloc[i] = np.random.uniform(left, right)
    return newx


def preprocess_ot_discrete(dat, protect):
    '''
    Remove dependencies via optimal transport on discrete data.

    This function removes the dependencies between the protected attribute (discrete)
    and the rest of the features (all discrete) via optimal transport.

    :param dat: A DataFrame or array-like of discrete data.
    :param protect: The column name of the protected attribute.

    :return: A DataFrame or array-like of data with the dependencies between
             the protected attribute and the rest of the features removed.
    '''
    modified_dat = dat.copy()
    tomodify = [col for col in dat.columns if col != protect]
    z = dat[protect]

    for col in tomodify:
        x = dat[col]
        newx = x.astype(float)  # Ensure column can handle float values
        ordered_condf = np.sort(x.unique())
        c_pmf = pd.crosstab(x, z)
        c_cmf = c_pmf.cumsum(axis=0).div(c_pmf.sum(axis=0), axis=1)

        for i in range(len(x)):
            xi_min = x[x < x[i]]
            xi_min = xi_min.max() if len(xi_min) > 0 else -1e12

            left = c_cmf.loc[xi_min, z.iloc[i]] if xi_min in c_cmf.index else 0
            right = c_cmf.loc[x[i], z.iloc[i]] if x[i] in c_cmf.index else 0

            newx.iloc[i] = np.random.uniform(left, right)

        modified_dat[col] = newx

    return modified_dat
