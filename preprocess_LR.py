import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def preprocess_lr(dat, protect):
    '''
    Creates an orthogonal representation of the data independent of the protected variable
    by replacing each variable with its residual from linear regression.

    :param dat: A DataFrame or array-like of the input data.
    :param protect: The column index or name of the protected variable.

    :return: A DataFrame or array-like of the data with each variable orthogonalized with respect to the protected variable.
    '''
    # Ensure protect is a column name
    if isinstance(protect, int):
        protect = dat.columns[protect]

    # Create a copy of the input data
    modified_dat = dat.copy()
    tomodify = [col for col in dat.columns if col != protect]

    # Add random noise ahead of regression if the protected variable is too sparse
    if np.std(dat[protect]) < 1e-6:
        dat[protect] += np.random.normal(size=len(dat))

    for i in tomodify:
        # Add random noise ahead of regression if the feature variable is too sparse
        if np.std(dat[i]) < 1e-6:
            dat[i] += np.random.normal(size=len(dat))

        # We will regress each non-protected variable x_i on the protected variable
        X = dat[[protect]].values  # Predictor (protected variable)
        y = dat[i].values  # Response (feature variable)

        # Fit ordinary least squares regression
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)

        # Replace the original feature with its residual if the regression is significant
        if model.score(X, y) > 0.01:
            modified_dat[i] = residuals

        # Add random noise if the residual variance is zero to avoid singularities
        if np.var(modified_dat[i]) == 0:
            modified_dat[i] += np.random.normal(size=len(modified_dat))

    return modified_dat
