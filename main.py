from experiments import getResultsPlot
import numpy as np
from UMFI import UMFI
import pandas as pd
from MCI import MCI
import matplotlib.pyplot as plt
import seaborn as sns


def plot_results(df):
    # Melt the dataframe for seaborn
    df_melted = df.melt(id_vars=["method"], var_name="Variable", value_name="Importance")

    # Create the box and whisker plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Variable", y="Importance", hue="method", data=df_melted)
    plt.xlabel("Variable")
    plt.ylabel("Variable Importance (%)")
    plt.title("Variable Importance by Method")
    plt.legend(title="Method")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    simTest = "Correlated_Interaction"
    nobs = 100
    niter = 100
    nX = 4
    results = getResultsPlot(simTest, nobs, niter, nX)
    results = getResultsPlot(simTest, nobs, niter, nX)
    print(results.head())

    # Plot the results
    plot_results(results)