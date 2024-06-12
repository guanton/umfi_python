import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_results(df, as_percentage=True, optimal=False, experiment_name=None):
    """
    Plots the UMFI feature importance scores.

    :param df: A DataFrame containing the feature importance scores with columns 'Feature', 'Importance', 'Method', 'Iteration', and 'Optimal'.
    :param as_percentage: bool, optional, default=True
                          If True, normalizes the UMFI scores so that they sum to 100% per method.
    :param experiment_name: str, optional, name of the experiment
    :param optimal: bool, optional, default=False
                    If True, plots only the scores from the optimal preprocessing method.
    """
    # Ensure the 'Importance' column is numeric
    df['Importance'] = pd.to_numeric(df['Importance'], errors='coerce')

    if optimal:
        # Filter for optimal methods only
        optimal_df = df[df['Optimal']]

        if as_percentage:
            # Normalize the importance scores to sum to 100% within the optimal scores
            optimal_df.loc[:, 'Importance'] = optimal_df.groupby(['Iteration'])['Importance'].apply(
                lambda x: 100 * x / x.sum()).reset_index(level=0, drop=True)

        plt.figure(figsize=(14, 7))
        sns.boxplot(x="Feature", y="Importance", hue="Method", data=optimal_df, palette="muted", showfliers=False)
        plt.xticks(rotation=90)
        plt.xlabel("Feature")
        if experiment_name is not None:
            plt.title(f"UMFI scores for {experiment_name} experiment")
        plt.ylabel("UMFI score" + (" (%)" if as_percentage else ""))
        plt.legend(title="Method")
        plt.show()
    else:
        if as_percentage:
            # Normalize the importance scores to sum to 100% per method and iteration
            df['Importance'] = df.groupby(['Method', 'Iteration'])['Importance'].apply(
                lambda x: 100 * x / x.sum()).reset_index(level=[0, 1], drop=True)

        # Convert 'Feature' and 'Method' to categorical if they are not already
        df['Feature'] = df['Feature'].astype('category')
        df['Method'] = df['Method'].astype('category')

        # Separate the DataFrame for different plots
        ot_df = df[df['Method'] == 'ot']
        lr_df = df[df['Method'] == 'lr']
        optimal_df = df[df['Optimal']]
        optimal_df = optimal_df.copy()
        optimal_df.loc[:, 'Method'] = 'Optimal'

        plt.figure(figsize=(14, 7))
        # Create the box and whisker plot for OT, LR, and Optimal methods
        sns.boxplot(x="Feature", y="Importance", hue="Method", data=pd.concat([ot_df, lr_df, optimal_df]), palette="muted", showfliers=False)
        plt.xticks(rotation=90)
        plt.xlabel("Feature")
        if experiment_name is not None:
            plt.title(f"UMFI scores for {experiment_name} experiment")
        plt.ylabel("UMFI Importance" + (" (%)" if as_percentage else ""))
        plt.legend(title="Method")
        plt.show()



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




