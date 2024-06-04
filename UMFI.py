import numpy as np
from preprocess_LR import preprocess_lr
from preprocess_OT import preprocess_ot, preprocess_ot_discrete
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

def UMFI(X, y, preprocessing="lr"):
    '''
    Calculate ultra-marginal feature importance.

    :param X: A DataFrame or array-like of shape (n_samples, n_features).
              The values for the predictors.
    :param y: A Series or array-like of shape (n_samples,).
              The values for the response.
    :param preprocessing: str, optional, default="lr"
                          Method for preprocessing the data.
                          Options are "lr" (Linear Regression) or "ot" (Optimal Transport).

    :return: A numpy array of shape (n_features,) containing the feature importance scores for each predictor.
    '''
    fi = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        col_name = X.columns[i]
        # print(f"Processing column: {col_name} with preprocessing: {preprocessing}")

        if preprocessing == "ot":
            if is_discrete(X[col_name]):
                newX = preprocess_ot_discrete(X, col_name)
            else:
                newX = preprocess_ot(X, col_name)
        elif preprocessing == "lr":
            newX = preprocess_lr(X, col_name)

        # Detect whether y is numeric or categorical
        if np.issubdtype(y.dtype, np.number):
            rf_with = RandomForestRegressor(n_estimators=100)
            rf_with.fit(newX, y)
            r2_with = max(r2_score(y, rf_with.predict(newX)), 0)

            newX_without = newX.drop(columns=[col_name])
            rf_without = RandomForestRegressor(n_estimators=100)
            rf_without.fit(newX_without, y)
            r2_without = max(r2_score(y, rf_without.predict(newX_without)), 0)

            fi[i] = r2_with - r2_without
        else:
            rf_with = RandomForestClassifier(n_estimators=100)
            rf_with.fit(newX, y)
            accuracy_with = max(accuracy_score(y, rf_with.predict(newX)), 0.5)

            newX_without = newX.drop(columns=[col_name])
            rf_without = RandomForestClassifier(n_estimators=100)
            rf_without.fit(newX_without, y)
            accuracy_without = max(accuracy_score(y, rf_without.predict(newX_without)), 0.5)

            fi[i] = accuracy_with - accuracy_without

        # print(f"Feature importance for column {col_name}: {fi[i]}")

    # Set negative feature importance scores to 0
    fi[fi < 0] = 0
    return fi




def is_discrete(column):
    '''
    Detect if the given column contains discrete values.

    :param column: Series or array-like
                   The column to check.

    :return: bool
             True if the column contains discrete values, otherwise False.
    '''
    # Check if all values are integers or can be converted to integers without loss
    return np.all(np.equal(np.mod(column, 1), 0))



