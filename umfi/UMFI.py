import numpy as np
import pandas as pd
from umfi.preprocess_LR import preprocess_lr
from umfi.preprocess_OT import preprocess_ot
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from tqdm import tqdm
from minepy import MINE


def UMFI(X, y, preprocessing_methods=["ot", "lr"], niter=10, n_trees=100, select_optimal_preprocessing=True, oob = True):
    """
    Calculate ultra-marginal feature importance for each feature in X with respect to response y.

    :param X: A DataFrame or array-like of shape (n_samples, n_features).
              The values for the predictors.
    :param y: A Series or array-like of shape (n_samples,).
              The values for the response.
    :param preprocessing_methods: list, optional, default=["ot", "lr"]
                                  Methods for preprocessing (dependency removal) the data for computing UMFI.
                                  Options are "lr" (Linear Regression) and "ot" (Optimal Transport).
    :param niter: int, optional, default=10
                  Number of iterations to calculate UMFI for each preprocessing method.
    :param n_trees: int, optional, default=100
                  Number of trees for random forest model
    :param select_optimal_preprocessing: bool, optional, default=False
                                         If True, UMFI is computed using the optimal preprocessing method (OT or LR) for
                                         each feature, based on the independence of the preprocessing with respect to
                                         the protected feature, and the amount of information removed
    :return:
    avg_importance_scores: A DataFrame containing the average feature importance scores for each predictor and method
    across all iterations
    final_df: A DataFrame containing the feature importance scores for each predictor, method, and iteration.
    """
    results = []

    for method in preprocessing_methods:
        for iteration in tqdm(range(niter), desc=f"Using {method} to remove dependencies"):
            umfi = np.zeros(X.shape[1])
            # initialize to compute maximal information coefficient and oob_r2_scores to measure quality of preprocessing
            if select_optimal_preprocessing:
                mic_scores = np.zeros(X.shape[1])
                oob_r2_scores = np.zeros(X.shape[1])

            for i in range(X.shape[1]):
                col_name = X.columns[i] # select current protected feature to compute its UMFI score
                # obtain the preprocessing, to remove dependencies of the protected feature from other features
                if method == "ot":
                    newX = preprocess_ot(X, col_name)
                elif method == "lr":
                    newX = preprocess_lr(X, col_name)
                if select_optimal_preprocessing:
                    # Compute maximal information coefficient (MIC) for each column
                    X_without = X.drop(columns=[col_name])
                    newX_without = newX.drop(columns=[col_name])
                    mic_total = 0

                    for col in X_without.columns:
                        mine = MINE(alpha=0.6, c=15, est="mic_approx")
                        mine.compute_score(X_without[col].to_numpy(), newX_without[col].to_numpy())
                        # print(f"Maximal Information Coefficient between raw {col} and {col} after preprocessing w/r/t {col_name}: {mine.mic()}")
                        mic_total += mine.mic()
                    mic_scores[i] = mic_total / X_without.shape[1]

                    # Compute OOB R² for the preprocessed data predicting the protected feature
                    rf_preprocess = RandomForestRegressor(n_estimators=n_trees, oob_score=True)
                    rf_preprocess.fit(newX_without, X[col_name])
                    oob_r2_preprocess = max(0, rf_preprocess.oob_score_)
                    oob_r2_scores[i] = oob_r2_preprocess
                    # print(f'OOB R^2 for preprocessing predicting feature {col_name}:', oob_r2_preprocess)


                if np.issubdtype(y.dtype, np.number):
                    rf_with = RandomForestRegressor(n_estimators=n_trees, oob_score=True)
                    rf_with.fit(newX, y)  # use preprocessed set along with the current feature to predict y
                    newX_without = newX.drop(columns=[col_name])
                    rf_without = RandomForestRegressor(n_estimators=n_trees, oob_score=True)
                    rf_without.fit(newX_without, y)  # use preprocessed set (without current feature) to predict y
                    if oob:
                        r2_with = max(0, rf_with.oob_score_) #max(r2_score(y, rf_with.predict(newX)), 0)
                        # print('OOB with:', r2_with)
                        r2_without = max(0, rf_without.oob_score_)  # max(r2_score(y, rf_without.predict(newX_without)), 0)
                        # print('OOB without:', r2_without)
                        # r2_with_ = max(r2_score(y, rf_with.predict(newX)), 0)
                        # print('overfit with:', r2_with_)
                        # r2_without_ = max(r2_score(y, rf_without.predict(newX_without)), 0)
                        # print('overfit without:', r2_without_)
                    else:
                        r2_with = max(r2_score(y, rf_with.predict(newX)), 0)
                        r2_without = max(r2_score(y, rf_without.predict(newX_without)), 0)
                    umfi[i] = r2_with - r2_without
                else:
                    rf_with = RandomForestClassifier(n_estimators=n_trees, oob_score = True)
                    rf_with.fit(newX, y)
                    newX_without = newX.drop(columns=[col_name])
                    rf_without = RandomForestClassifier(n_estimators=n_trees, oob_score=True)
                    rf_without.fit(newX_without, y)
                    if oob:
                        accuracy_with = max(0, rf_with.oob_score_)#max(accuracy_score(y, rf_with.predict(newX)), 0.5)
                        accuracy_without = max(0, rf_without.oob_score_)#max(accuracy_score(y, rf_without.predict(newX_without)), 0.5)
                        # print('OOB with:', accuracy_with)
                        # print('OOB without:', accuracy_without)
                    else:
                        accuracy_with = max(accuracy_score(y, rf_with.predict(newX)), 0.5)
                        accuracy_without = max(accuracy_score(y, rf_without.predict(newX_without)), 0.5)
                    umfi[i] = accuracy_with - accuracy_without

            # Set negative feature importance scores to 0
            umfi[umfi < 0] = 0

            # Create a DataFrame for the current method and iteration
            method_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': umfi,
                'Method': method,
                'MIC': mic_scores if select_optimal_preprocessing else None,
                'OOB_R2': oob_r2_scores if select_optimal_preprocessing else None,
                'Iteration': iteration,
                'Optimal': None  # Placeholder for optimal method indicator
            })

            results.append(method_df)

    # Concatenate results for all methods
    final_df = pd.concat(results, ignore_index=True)

    if select_optimal_preprocessing:
        for feature in X.columns:
            feature_df = final_df[final_df['Feature'] == feature].copy()
            feature_df.loc[:, 'Metric'] = feature_df['MIC'] + 10 * (1 - feature_df['OOB_R2'])
            best_method = feature_df.loc[feature_df['Metric'].idxmax(), 'Method']
            final_df.loc[final_df['Feature'] == feature, 'Optimal'] = final_df['Method'] == best_method

        optimal_scores = final_df[final_df['Optimal']]
        avg_importance_scores = optimal_scores.groupby('Feature')['Importance'].mean().reset_index()
        return avg_importance_scores, final_df
    else:
        avg_importance_scores = final_df.groupby(['Feature', 'Method'])['Importance'].mean().reset_index()
        return avg_importance_scores, final_df


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



