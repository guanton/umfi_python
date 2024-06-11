import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def is_discrete(series):
    return pd.api.types.is_integer_dtype(series)


def plot_results(df, as_percentage=False):
    '''
    Plots the UMFI feature importance scores.

    :param df: A DataFrame containing the feature importance scores with columns 'Feature', 'Importance', 'Method', and 'Iteration'.
    :param as_percentage: bool, optional, default=False
                          If True, normalizes the UMFI scores so that they sum to 100% per method.
    '''
    # Ensure the 'Importance' column is numeric
    df['Importance'] = pd.to_numeric(df['Importance'], errors='coerce')

    if as_percentage:
        # Normalize the importance scores to sum to 100% per method
        df['Importance'] = df.groupby(['Method', 'Iteration'])['Importance'].apply(lambda x: 100 * x / x.sum()).reset_index(level=[0, 1], drop=True)

    # Convert 'Feature' and 'Method' to categorical if they are not already
    df['Feature'] = df['Feature'].astype('category')
    df['Method'] = df['Method'].astype('category')

    # Create the box and whisker plot
    plt.figure(figsize=(14, 7))
    sns.boxplot(x="Feature", y="Importance", hue="Method", data=df)
    plt.xticks(rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("UMFI Importance" + (" (%)" if as_percentage else ""))
    plt.title("UMFI Feature Importance")
    plt.legend(title="Method")
    plt.show()


