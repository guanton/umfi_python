import pandas as pd
import numpy as np
import statsmodels.api as sm

def preprocess_lr(dat, protect):
    # Create a copy of the input data
    modifiedDAT = dat.copy()
    tomodify = [col for col in range(dat.shape[1]) if col != protect]

    # add random noise ahead of regression if data is too sparse
    if np.std(dat.iloc[:, protect]) < 1e-6:
        dat.iloc[:, protect] += np.random.normal(size=len(dat))

    for i in tomodify:
        # add random noise ahead of regression if data is too sparse
        if np.std(dat.iloc[:, i]) < 1e-6:
            dat.iloc[:, i] += np.random.normal(size=len(dat))

        # we will regress each non-protected variable x_i on the protected variable
        X = sm.add_constant(dat.iloc[:, protect]) # this adds an intercept term to the predictor
        # X = dat.iloc[:, protect]
        y = dat.iloc[:, i]

        # fit ordinary least squares regression
        model = sm.OLS(y, sm.add_constant(X)).fit()

        if model.pvalues[1] < 0.01:
            modifiedDAT.iloc[:, i] = model.resid

        if modifiedDAT.iloc[:, i].var() == 0:
            modifiedDAT.iloc[:, i] = np.random.normal(size=len(modifiedDAT))

    return modifiedDAT
