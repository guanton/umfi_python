import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def preprocess_lr(dat, protect):
    '''
    This function preprocess the feature set (dat) to remove dependencies between the protected feature (Z)
    and the rest of the features (X) by replacing each variable with its residual from linear regression

    :param dat: A DataFrame or array-like of the input features
    :param protect: The column index or name of the protected variable.

    :return: A DataFrame or array-like of the data with each variable orthogonalized with respect to the protected variable.
    '''
    # Ensure protect is a column name
    if isinstance(protect, int):
        protect = dat.columns[protect]

    # Create a copy of the input data
    modified_dat = dat.copy()
    # we will remove dependencies of the protected feature from all other features
    tomodify = [col for col in dat.columns if col != protect]

    # Add random noise ahead of regression if the protected variable is too sparse
    if np.std(dat[protect]) < 1e-6:
        dat[protect] += np.random.normal(size=len(dat))

    X = dat[[protect]].values  # Predictor (protected variable)

    for col in tomodify:
        # Add random noise ahead of regression if the feature variable is too sparse
        if np.std(dat[col]) < 1e-6:
            dat[col] += np.random.normal(size=len(dat))

        y = dat[col].values  # Response (feature variable)

        # Fit ordinary least squares regression
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)

        # Replace the original feature with its residual if the regression is significant
        if model.score(X, y) > 0.01:
            modified_dat[col] = residuals

        # Add random noise if the residual variance is zero to avoid singularities
        if np.var(modified_dat[col])  < 1e-10:
            modified_dat[col] = np.random.normal(size=len(modified_dat))

    return modified_dat
