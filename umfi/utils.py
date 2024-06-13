import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_results(df, as_percentage=True, experiment_name=None):
    """
    Plots the UMFI feature importance scores.

    :param df: A DataFrame containing the feature importance scores with columns 'Feature', 'Importance', 'Preprocessing', 'Iteration', and 'Selected_Method'.
    :param as_percentage: bool, optional, default=True
                          If True, normalizes the UMFI scores so that they sum to 100% per method.
    :param experiment_name: str, optional, name of the experiment
    """
    # Ensure the 'Importance' column is numeric
    df['Importance'] = pd.to_numeric(df['Importance'], errors='coerce')

    if as_percentage:
        # Normalize the importance scores to sum to 100% per method and iteration
        df['Importance'] = df.groupby(['Preprocessing', 'Iteration'])['Importance'].apply(
            lambda x: 100 * x / x.sum()).reset_index(level=[0, 1], drop=True)

    # Convert 'Feature' and 'Method' to categorical if they are not already
    df['Feature'] = df['Feature'].astype('category')
    df['Method'] = df['Preprocessing'].astype('category')

    plt.figure(figsize=(14, 7))
    # Create the box and whisker plot for all methods present in the DataFrame
    sns.boxplot(x="Feature", y="Importance", hue="Preprocessing", data=df, palette="muted", showfliers=False)
    plt.xticks(rotation=90)
    plt.xlabel("Feature")
    if experiment_name is not None:
        plt.title(f"UMFI scores for {experiment_name} experiment")
    plt.ylabel("UMFI Importance" + (" (%)" if as_percentage else ""))
    plt.legend(title="Preprocessing method")
    plt.savefig(f"{experiment_name}_UMFI_scores.png")
    plt.show()
    plt.close()



def is_discrete(series, threshold=0.05):
    """
    Determines if a series is likely from a discrete random variable.

    :param series: The pandas Series to check.
    :param threshold: The proportion of unique values to total values below which the series is considered discrete.
    :return: True if the series is likely discrete, False otherwise.
    """
    if pd.api.types.is_integer_dtype(series):
        return True
    else:
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        unique_values_ratio = series.nunique() / len(series)
    return unique_values_ratio < threshold




