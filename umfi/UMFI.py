import numpy as np
import pandas as pd
from umfi.preprocess_LR import preprocess_lr
from umfi.preprocess_OT import preprocess_ot
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from tqdm import tqdm
from minepy import MINE


def UMFI(X, y, preprocessing_methods=["ot"], niter=10, n_trees = 100):
    '''
    Calculate ultra-marginal feature importance.

    :param X: A DataFrame or array-like of shape (n_samples, n_features).
              The values for the predictors.
    :param y: A Series or array-like of shape (n_samples,).
              The values for the response.
    :param preprocessing_methods: list, optional, default=["ot"]
                                  Methods for preprocessing the data for computing UMFI.
                                  Options are "lr" (Linear Regression) or "ot" (Optimal Transport).
    :param niter: int, optional, default=10
                  Number of iterations to calculate UMFI for each preprocessing method.
    :param n_trees: int, optional, default=100
                  Number of trees for random forest model
    :return: A DataFrame containing the feature importance scores for each predictor, method, and iteration.
    '''
    results = []


    for method in preprocessing_methods:
        for _ in tqdm(range(niter), desc=f"Using {method} to remove dependencies"):
            umfi = np.zeros(X.shape[1])
            correlations = np.zeros(X.shape[1])
            # compute UMFI for each feature (not including response)
            for i in range(X.shape[1]):
                col_name = X.columns[i]
                # use specified preprocessing method to remove dependencies of the current feature from
                # the rest of the feature set
                if method == "ot":
                    newX = preprocess_ot(X, col_name)

                elif method == "lr":
                    newX = preprocess_lr(X, col_name)
                # Compute correlation between X and newX
                print(X.corrwith(newX))
                # Compute mutual information score for each column
                X_without = X.copy().drop(columns=[col_name])
                newX_without = newX.copy().drop(columns=[col_name])
                mine = MINE(alpha=0.6, c=15, est="mic_approx")

                for col in X_without.columns:
                    mine.compute_score(X_without[col].to_numpy(), newX_without[col].to_numpy())
                    print(f"Maximal Information Coefficient between raw {col} and {col} after preprocessing w/r/t {col_name}: {mine.mic()}")
                # correlations[i] = X.corrwith(newX).mean()
                # use preprocessed set to predict protected feature
                rf_preprocess = RandomForestRegressor(n_estimators=n_trees, oob_score=True)
                rf_preprocess.fit(newX_without, X[col_name])
                oob_r2_preprocess = max(0, rf_preprocess.oob_score_)

                print(f'OOB R^2 for preprocessing predicting feature {col_name}:', oob_r2_preprocess)

                # Detect whether y is numeric or categorical
                if np.issubdtype(y.dtype, np.number):
                    rf_with = RandomForestRegressor(n_estimators=n_trees)
                    rf_with.fit(newX, y) # use preprocessed set along with the current feature to predict y
                    r2_with = max(r2_score(y, rf_with.predict(newX)), 0)
                    newX_without = newX.drop(columns=[col_name])
                    rf_without = RandomForestRegressor(n_estimators=n_trees)
                    rf_without.fit(newX_without, y) # use preprocessed set (without current feature) to predict y
                    r2_without = max(r2_score(y, rf_without.predict(newX_without)), 0)
                    umfi[i] = r2_with - r2_without
                else:
                    rf_with = RandomForestClassifier(n_estimators=n_trees)
                    rf_with.fit(newX, y)
                    accuracy_with = max(accuracy_score(y, rf_with.predict(newX)), 0.5)

                    newX_without = newX.drop(columns=[col_name])
                    rf_without = RandomForestClassifier(n_estimators=n_trees)
                    rf_without.fit(newX_without, y)
                    accuracy_without = max(accuracy_score(y, rf_without.predict(newX_without)), 0.5)

                    umfi[i] = accuracy_with - accuracy_without

            # Set negative feature importance scores to 0
            umfi[umfi < 0] = 0

            # Create a DataFrame for the current method and iteration
            method_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': umfi,
                'Method': method,
                'Correlation': correlations,
                'Iteration': _
            })

            results.append(method_df)

    # Concatenate results for all methods
    final_df = pd.concat(results, ignore_index=True)
    return final_df

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



