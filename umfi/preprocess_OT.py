import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from ot import emd
from umfi.experiments import *
from umfi.utils import is_discrete
from scipy.optimize import linear_sum_assignment

def preprocess_ot(dat, protect, n_quan=None, min_sd=1e-6, discretization_quantiles = 5):
    '''
    This function iteratively preprocesses
    Preprocess the data via the optimal transport solution F_X|Z

    This function removes dependencies between the protected attribute (continuous)
    and the rest of the features via optimal transport or linear regression.

    :param dat: A DataFrame or array-like of continuous data.
    :param protect: The column name of the protected attribute.
    :param n_quan: The number of quantiles to use to estimate the conditional CDF (default is ceil(n_samples/150)).
    :param min_sd: The minimum standard deviation that data points within a quantile can have before noise is added (default is 1e-6)
    :param discretization_quantiles: The number of quantiles to discretize a continuous r.v. into if applicable
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
            # if x is discrete and z is cts, then we should discretize z
            if not is_discrete(z):
                ranks = z.argsort().argsort()
                z = np.round(ranks * discretization_quantiles / len(z),0).astype(int)
            newx = discrete_univariate_optimal_transport(x, z)
        else:
            # if x is cts and z is discrete, then we should discretize x
            if is_discrete(z):
                ranks = x.argsort().argsort()
                x = np.round(ranks * discretization_quantiles / len(z), 0).astype(int)
                newx = discrete_univariate_optimal_transport(x, z)
            else:
                x = dat[j].values  # Extract the feature to modify as a NumPy array
                newx = x.copy()  # Create a copy of the feature
                for quan in range(1, n_quan):

                    cur_obs = (z <= min(quans[quan], max(z))) & (z > quans[quan - 1])
                    # print(f"count for quantile {quan}: count = {np.sum(cur_obs)}")
                    x_curquan = x[cur_obs]
                    z_curquan = z[cur_obs]
                    # plt.scatter(x_curquan, z_curquan)
                    # plt.xticks(rotation=90)
                    # plt.xlabel("x_curquan")
                    # plt.ylabel("z_curquan")
                    # plt.show()
                    # break

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
                    # slope = model.coef_[0]
                    # print('slope:', slope)
                    # Calculate conditional CDF
                    rv = x_curquan - model.predict(z_curquan.reshape(-1, 1)) # residuals
                    newx[cur_obs] = (rv.argsort().argsort() + 1)  #
        modified_dat[j] = newx

    return modified_dat







def discrete_univariate_optimal_transport(x, z):
    """
    Applies discrete transformation to the data using cross-tabulation and cumulative marginal frequencies.

    :param x: Feature to be transformed.
    :param z: Protected attribute.
    :return: Transformed feature.
    """
    newx = x.astype(float)
    c_pmf = pd.crosstab(x, z)

    c_cmf = c_pmf.cumsum(axis=0).div(c_pmf.sum(axis=0), axis=1)
    # print(c_cmf.head())
    for i in range(len(x)):
        xi_min = x[x < x[i]]
        if len(xi_min) > 0:
            xi_min = xi_min.max()
        else:
            xi_min = -np.inf  # Handle the case where there's no such value

        l_xi = c_cmf.loc[xi_min, z[i]] if xi_min in c_cmf.index else 0
        r_xi = c_cmf.loc[x[i], z[i]] if x[i] in c_cmf.index else 0

        u_i = np.random.uniform(l_xi, r_xi)
        newx.iloc[i] = u_i
    # if is_discrete(z):
    #
    # else:
    #     # Handling the case where Z is continuous
    #     for i in range(len(x)):
    #         xi_min = x[x < x[i]]
    #         if len(xi_min) > 0:
    #             xi_min = xi_min.max()
    #         else:
    #             xi_min = -np.inf  # Handle the case where there's no such value
    #
    #         # Calculate the empirical CDF for the current value of Z
    #         z_i = z[i]
    #         cdf_z = (z <= z_i).mean()
    #
    #         # Determine l_xi and r_xi based on the empirical CDF
    #         l_xi = (x[x == xi_min] <= z_i).mean() if xi_min != -np.inf else 0
    #         r_xi = (x[x == x[i]] <= z_i).mean()
    #
    #         u_i = np.random.uniform(l_xi, r_xi)
    #         newx.iloc[i] = u_i
    #
    # return newx

    return newx

if __name__ == "__main__":
    nobs = 1000
    x0 = np.random.normal(0, 1, nobs)
    x1 = np.random.uniform(-0.5, 0.5, nobs)

    # Generate variables x1 and S from a standard normal distribution N(0, 1)
    x2 = np.random.normal(0, 1, nobs)
    x3 = np.random.binomial(4, 0.5, nobs) - 2
    x4 = np.random.binomial(4, 0.5, nobs)
    # Calculate Y based on the condition
    y = 10 * np.sign(x0 * x3) + 2 * x2 + x4

    # Create a DataFrame with the generated data
    data = {
        'x0': x0,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'y': y
    }
    dat = pd.DataFrame(data)
    new_x = preprocess_ot(dat, 'x0')

    new_x3 = discrete_transform(dat['x3'], x0)



