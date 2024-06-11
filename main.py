from umfi.experiments import *
import numpy as np
from umfi.UMFI import UMFI
import pandas as pd
from umfi.utils import plot_results
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadr
from tqdm import tqdm


def calculate_feature_importance(df, methods, niter):
    """
    Calculates feature importance for different methods.

    :param df: A DataFrame containing the data.
    :param methods: A dictionary of methods for preprocessing.
    :param niter: Number of iterations.
    :return: A DataFrame containing the feature importance scores.
    """
    nX = df.shape[1] - 1 # number of features (not including response)
    Imp = {method: np.zeros((niter, nX)) for method in methods.keys()}

    for i in tqdm(range(niter)):
        for method, preprocess_func in methods.items():
            Imp[method][i, :] = UMFI(df.iloc[:, :-1], df['y'], preprocessing=preprocess_func)

    # Normalize feature importance
    Imp2 = Imp.copy()
    epsilon = 1e-8  # Small constant to prevent division by zero
    for key, value in Imp.items():
        row_sums = value.sum(axis=1).reshape(-1, 1)
        row_sums[row_sums == 0] = epsilon  # Replace zero sums with epsilon
        Imp2[key] = (value * 100) / row_sums

    # Convert results to data frames and concatenate properly
    results = []
    for key, value in Imp2.items():
        df_result = pd.DataFrame(value, columns=df.columns[:-1])
        df_result['method'] = key
        results.append(df_result)

    final_df = pd.concat(results, ignore_index=True)
    return final_df
# def plot_results(df):
#     # Melt the dataframe for seaborn
#     df_melted = df.melt(id_vars=["method"], var_name="Variable", value_name="Importance")
#
#     # Create the box and whisker plot
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x="Variable", y="Importance", hue="method", data=df_melted)
#     plt.xlabel("Variable")
#     plt.ylabel("Variable Importance (%)")
#     plt.title("Variable Importance by Method")
#     plt.legend(title="Method")
#     plt.show()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    nobs = 1000
    niter = 5
    # simTest = generate_msq_data(nobs)
    BRCA_data = pyreadr.read_r('data/BRCA.rda')  # Adjust the path as needed
    # Assuming the .rda file contains a single dataframe
    BRCA_df = list(BRCA_data.values())[0]
    # print(BRCA_df.head())

    # Generate data
    data = generate_mixed_cts_discrete_data(nobs)#generate_terc2_data(nobs)
    # Separate features and target
    X = data.drop(columns=['y'])
    y = data['y']

    # Calculate ultra-marginal feature importance using both preprocessing methods
    results = UMFI(X, y, preprocessing_methods=["ot", "lr"], niter = niter)
    results.head()

    # Plot the results
    plot_results(results,as_percentage=True)
    #generate_correlated_interaction_data(nobs)
    # generate_blood_relation_data(nobs)
    # generate_svq_data(nobs)
    # generate_rvq_data(nobs)
    # generate_blood_relation_data(nobs)
    # generate_correlation_data(nobs)
    # generate_nonlinear_interaction_data(nobs)
    # generate_correlated_interaction_data(nobs)





    # # Define feature importance methods that will be compared
    # methods = {
    #     "UMFI_LR": "lr",
    #     "UMFI_OT": "ot"
    # }

    # # Calculate feature importance
    # fi_lr = UMFI(X, y, preprocessing='lr')
    # fi_ot = UMFI(X, y, preprocessing='ot')

    # # Calculate feature importance
    # results = calculate_feature_importance(simTest, methods, niter)
    #
    # # Plot the results
    # plot_results(results)
    #
    # importance_df = pd.DataFrame({
    #     'Feature': X.columns,
    #     'UMFI_LR': fi_lr,
    #     'UMFI_OT': fi_ot
    # })
    #
    # # Plot the results
    # plot_results(importance_df)