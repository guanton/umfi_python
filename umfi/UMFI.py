import numpy as np
import pandas as pd
from umfi.preprocess_LR import preprocess_lr
from umfi.preprocess_OT import preprocess_ot
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
from minepy import MINE


def UMFI(X, y, preprocessing_methods=["optimal transport", "linear regression"], niter=10, n_trees=100, select_optimal_preprocessing=True, RF = False, dependence_metric = "OOB_R2"):
    """
    Calculate ultra-marginal feature importance for each feature in X with respect to response y.

    :param X: A DataFrame or array-like of shape (n_samples, n_features).
              The values for the predictors.
    :param y: A Series or array-like of shape (n_samples,).
              The values for the response.
    :param preprocessing_methods: list, optional, default=["optimal transport", "linear regression"]
                                  Methods for preprocessing (dependency removal) the data for computing UMFI.
                                  Options are "lr" (Linear Regression) and "ot" (Optimal Transport).
    :param niter: int, optional, default=10
                  Number of iterations to calculate UMFI for each preprocessing method.
    :param n_trees: int, optional, default=100
                  Number of trees for random forest model
    :param select_optimal_preprocessing: bool, optional, default=False
                                         If True, UMFI is computed using the optimal preprocessing method (OT or LR) for
                                         each feature, based on the independence of the preprocessing with respect to
                                         the protected feature, and the amount of information removed.
    :param  RF: bool, optional, default= False,
                If True, all models are fitted with random forest. If False, we use extra trees
    :param dependence_metric: str, optional, default = 'both'
                            If 'MI': we only use pairwise mutual information to measure the independence of the preprocessing
                            with respect to the protected feature
                            If 'OOB_R2': we use the out of bag R^2 value of the preprocessing for predicting the protected
                            feature to measure independence
                            If 'both': we use the sum
    :return:
    avg_importance_scores: A DataFrame containing the average feature importance scores for each predictor and method
    across all iterations
    final_df: A DataFrame containing the feature importance scores for each predictor, preprocessing method, and iteration.
    """
    results = []

    for method in preprocessing_methods:
        for iteration in tqdm(range(niter), desc=f"Using {method} to remove dependencies"):
            umfi = np.zeros(X.shape[1])
            # initialize to compute maximal information coefficient and oob_r2_scores to measure quality of preprocessing
            if select_optimal_preprocessing:
                mic_scores = np.zeros(X.shape[1])
                ind_scores = np.zeros(X.shape[1])

            for i in range(X.shape[1]):
                col_name = X.columns[i]  # select current protected feature to compute its UMFI score
                # obtain the preprocessing, to remove dependencies of the protected feature from other features
                if method == "optimal transport":
                    newX = preprocess_ot(X, col_name)
                elif method == "linear regression":
                    newX = preprocess_lr(X, col_name)
                # these represent the raw data and preprocessed data without the current protected feature
                X_without = X.drop(columns=[col_name])
                newX_without = newX.drop(columns=[col_name])
                if select_optimal_preprocessing:
                    # quantify the information preservation via the maximal information coefficient (MIC)
                    # across each transformed variable (averaged)
                    mic_total = 0
                    for col in X_without.columns:
                        mine = MINE(alpha=0.6, c=15, est="mic_approx")
                        mine.compute_score(X_without[col].to_numpy(), newX_without[col].to_numpy())
                        mic_total += mine.mic()
                    mic_scores[i] = mic_total / X_without.shape[1]
                    # quantify the dependence of the preprocessing on the protected feature
                    # using either OOB_R2 and/or mutual_info_regression
                    if dependence_metric == 'OOB_R2' or 'both':
                        if np.issubdtype(X[col_name].dtype, np.number):
                            if RF:
                                indep_model = RandomForestRegressor(n_estimators=n_trees, oob_score=True)
                            else:
                                indep_model = ExtraTreesRegressor(n_estimators=n_trees, oob_score=True, bootstrap=True)
                        else:
                            if RF:
                                indep_model = RandomForestClassifier(n_estimators=n_trees, oob_score=True)
                            else:
                                indep_model = ExtraTreesClassifier(n_estimators=n_trees, oob_score = True, bootstrap=True)
                        indep_model.fit(newX_without, X[col_name])
                        oob_r2_preprocess = max(0, indep_model.oob_score_)
                        if dependence_metric != 'both':
                            ind_scores[i] = oob_r2_preprocess
                    if dependence_metric == 'MI' or 'both':
                        mutual_info = mutual_info_regression(newX_without, X[col_name])
                        if dependence_metric != 'both':
                            ind_scores[i] = np.mean(mutual_info)
                    if dependence_metric == 'both':
                        ind_scores[i] = np.mean(mutual_info) + oob_r2_preprocess
                # fit appropriate models
                if np.issubdtype(y.dtype, np.number):
                    if RF:
                        model_with = RandomForestRegressor(n_estimators=n_trees, oob_score=True)
                        model_without = RandomForestRegressor(n_estimators=n_trees, oob_score=True)
                    else:
                        model_with = ExtraTreesRegressor(n_estimators=n_trees, oob_score=True, bootstrap=True)
                        model_without = ExtraTreesRegressor(n_estimators=n_trees, oob_score=True, bootstrap=True)
                else:
                    if RF:
                        model_with = RandomForestClassifier(n_estimators=n_trees, oob_score=True)
                        model_without = RandomForestClassifier(n_estimators=n_trees, oob_score=True)
                    else:
                        model_with = ExtraTreesClassifier(n_estimators=n_trees, oob_score=True, bootstrap=True)
                        model_without = ExtraTreesClassifier(n_estimators=n_trees, oob_score=True, bootstrap=True)
                # use models to compute UMFI
                model_with.fit(newX, y)  # use preprocessed set along with the current feature to predict y
                model_without.fit(newX_without, y)  # use preprocessed set (without current feature) to predict y
                score_with = max(0, model_with.oob_score_)
                score_without = max(0, model_without.oob_score_)
                umfi[i] = score_with - score_without


            # Set negative feature importance scores to 0
            umfi[umfi < 0] = 0

            # Create a DataFrame for the current method and iteration
            method_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': umfi,
                'Preprocessing': method,
                'MIC': mic_scores if select_optimal_preprocessing else None,
                'Independence': ind_scores if select_optimal_preprocessing else None,
                'Iteration': iteration,
            })

            results.append(method_df)

    # Concatenate results for all methods
    final_df = pd.concat(results, ignore_index=True)

    # we do another pass if we want to optimize UMFI by choosing the best preprocessing method for each feature
    if select_optimal_preprocessing:
        optimal_results = []
        for iteration in range(niter):
            iteration_df = final_df[final_df['Iteration'] == iteration]
            for feature in X.columns:
                feature_df = iteration_df[iteration_df['Feature'] == feature].copy()
                feature_df.loc[:, 'Metric'] = (1 - feature_df['MIC']) + 2 * feature_df['Independence']
                best_method = feature_df.loc[feature_df['Metric'].idxmin(), 'Preprocessing']
                optimal_importance = feature_df.loc[feature_df['Metric'].idxmin(), 'Importance']
                optimal_row = pd.DataFrame({
                    'Feature': [feature],
                    'Importance': [optimal_importance],
                    'Preprocessing': ['pick best'],
                    'Selected_Method': [best_method],
                    'MIC': [feature_df.loc[feature_df['Metric'].idxmin(), 'MIC']],
                    'Independence': [feature_df.loc[feature_df['Metric'].idxmin(), 'Independence']],
                    'Iteration': [iteration]
                })
                optimal_results.append(optimal_row)

        optimal_df = pd.concat(optimal_results, ignore_index=True)
        final_df = pd.concat([final_df, optimal_df], ignore_index=True)

        optimal_scores = final_df[final_df['Preprocessing'] == 'pick best']
        avg_importance_scores = optimal_scores.groupby('Feature')['Importance'].mean().reset_index()
        return avg_importance_scores, final_df
    else:
        avg_importance_scores = final_df.groupby(['Feature', 'Preprocessing'])['Importance'].mean().reset_index()
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



