from umfi.experiments import *
import numpy as np
from umfi.UMFI import UMFI
import matplotlib.pyplot as plt
import pandas as pd
from umfi.utils import plot_results

def run_experiment(experiment_name, nobs, niter, as_percentage = True, select_optimal_preprocessing=True):
    """
    :param experiment_name: name of experiment
    :param nobs: number of observations for simulated data
    :param niter: number of iterations to run UMFI scores
    :param as_percentage: boolean, if true, normalizes UMFI scores in plot as percentages
    :param select_optimal_preprocessing: boolean, if true, the average UMFI scores will automatically use the best
    preprocessing in the computation. These scores will be reported in avg_scores
    :return:
    """
    if experiment_name == 'nonlinear_interaction':
        data  = generate_nonlinear_interaction_data(nobs)
    if experiment_name == 'correlation':
        data = generate_correlation_data(nobs)
    if experiment_name == 'correlated_interaction':
        data = generate_correlated_interaction_data(nobs)
    if experiment_name == 'blood_relation':
        data = generate_blood_relation_data(nobs)
    if experiment_name == 'terc1':
        data = generate_terc1_data(nobs)
    if experiment_name == 'terc2':
        data = generate_terc2_data(nobs)
    if experiment_name == 'rvq':
        data = generate_rvq_data(nobs)
    if experiment_name == 'svq':
        data = generate_svq_data(nobs)
    if experiment_name == 'msq':
        data = generate_msq_data(nobs)
    if experiment_name == 'sg':
        data = generate_sg_data(nobs)
    if experiment_name == 'mixed_cts_discrete':
        data = generate_mixed_cts_discrete_data(nobs)
    if experiment_name == 'CAMELS':
        data = obtain_CAMELS_data()
    if experiment_name == 'BRCA':
        data = obtain_BRCA_data()
        response_col_name = 'BRCA_Subtype_PAM50'
    else:
        response_col_name = 'y'
    X = data.drop(columns=[response_col_name])
    y = data[response_col_name]
    avg_scores, detailed_results = UMFI(X, y, preprocessing_methods=["optimal transport"], niter=niter,
                                        select_optimal_preprocessing=False)
    print(f"Results for {experiment_name}: ")
    print(avg_scores)
    print(detailed_results)
    # Save avg_scores and detailed_results as JSON files
    avg_scores.to_json(f"{experiment_name}_avg_scores_niter-{niter}.json", orient='records', lines=True)
    detailed_results.to_json(f"{experiment_name}_detailed_results_niter-{niter}.json", orient='records', lines=True)
    # Plot the results
    plot_results(detailed_results, as_percentage = as_percentage, experiment_name = experiment_name)



if __name__ == "__main__":
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Ensure the display is wide enough
    pd.set_option('display.max_colwidth', None)  # Show full content of each column
    nobs = 1000
    niter = 50
    simulated_data_experiments = ["terc1", "terc2", "rvq", "svq", "msq", "sg"]
    for experiment_name in simulated_data_experiments:
        run_experiment(experiment_name, nobs, niter)




