import pandas as pd
import numpy as np
import statsmodels.api as sm

def preprocess_ot(dat, protect, n_quan=None, min_sd=None):
    if n_quan is None:
        n_quan = np.ceil(dat.shape[0] / 150).astype(int)
    if min_sd is None:
        min_sd = 1e-6

    modifiedDAT = dat.copy()  # Create a copy of the input data
    tomodify = [col for col in dat.columns if col != protect]
    z = dat[protect].values  # Extract the protected attribute as a NumPy array
    quans = np.linspace(0, 1, n_quan)  # Quantiles of interest
    quans = np.quantile(z, quans)  # Calculate quantiles of the protected attribute

    for j in tomodify:
        x = dat[j].values  # Extract the feature to modify as a NumPy array
        newx = x.copy()  # Create a copy of the feature
        for quan in range(1, n_quan):
            cur_obs = np.logical_and(z <= quans[quan], z >= quans[quan - 1])

            x_curquan = x[cur_obs]
            z_curquan = z[cur_obs]

            if x_curquan.std() < min_sd:
                x_curquan += np.random.normal(scale=x.std() / len(x), size=len(x_curquan))

            if z_curquan.std() < min_sd:
                z_curquan += np.random.normal(scale=z.std() / len(z), size=len(z_curquan))

            # Fit a linear regression model to approximate the conditional distribution
            X = sm.add_constant(z_curquan)  # Add a constant for the intercept
            model = sm.OLS(x_curquan, X).fit()

            # Calculate conditional CDF
            rv = model.resid
            condF = np.argsort(rv) / len(rv)
            newx[cur_obs] = condF

        modifiedDAT[j] = newx

    return modifiedDAT
