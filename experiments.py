import pandas as pd
import numpy as np

def generate_nonlinear_interaction_data(nobs):
    """
    Generates data for a nonlinear interaction scenario.

    :param nobs: Number of observations.
    :return: A DataFrame containing the generated data.
    """
    # Generate the variables x1, x2, x3, and x4 from a standard normal distribution
    x1 = np.random.normal(0, 1, nobs)
    x2 = np.random.normal(0, 1, nobs)
    x3 = np.random.normal(0, 1, nobs)
    x4 = np.random.normal(0, 1, nobs)

    # Calculate y according to the given equation
    y = x1 + x2 + np.sign(x1 * x2) + x3 + x4

    # Create a DataFrame with the generated data
    data = {
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'y': y
    }

    return pd.DataFrame(data)

def generate_correlated_interaction_data(nobs):
    """
    Generates data for the 'Correlated_Interaction' scenario.

    :param nobs: Number of observations.
    :param nvars: Number of variables.
    :return: A DataFrame containing the generated data.
    """
    A = np.random.normal(0, 1, nobs)
    B = np.random.normal(0, 1, nobs)
    C = np.random.normal(0, 1, nobs)
    D = np.random.normal(0, 1, nobs)
    E = np.random.normal(0, 1, nobs)
    G = np.random.normal(0, 1, nobs)

    data = {
        'x1': A + B,
        'x2': B + C,
        'x3': D + E,
        'x4': E + G
    }
    data['y'] = data['x1'] + data['x2'] + np.sign(data['x1'] * data['x2']) + data['x3'] + data['x4']

    return pd.DataFrame(data)


def generate_correlation_data(nobs):
    """
    Generates data for a correlated scenario.

    :param nobs: Number of observations.
    :return: A DataFrame containing the generated data.
    """
    # Generate the variables x1, x2, x4 from a standard normal distribution N(0, 1)
    x1 = np.random.normal(0, 1, nobs)
    x2 = np.random.normal(0, 1, nobs)
    x4 = np.random.normal(0, 1, nobs)

    # Generate epsilon from a normal distribution N(0, 0.01)
    epsilon = np.random.normal(0, 0.01, nobs)

    # Calculate x3 based on the given relationship
    x3 = x1 + epsilon

    # Calculate y based on the given relationship
    y = x1 + x2

    # Create a DataFrame with the generated data
    data = {
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'y': y
    }

    return pd.DataFrame(data)


def generate_blood_relation_data(nobs):
    """
    Generates data for a 'Blood Relation' scenario.

    :param nobs: Number of observations.
    :return: A DataFrame containing the generated data.
    """
    # Generate random variables
    delta = np.random.uniform(-1, 1, nobs)
    gamma = np.random.exponential(1, nobs)
    epsilon = np.random.uniform(-0.5, 0.5, nobs)

    # Generate variables x1 and S from a standard normal distribution N(0, 1)
    x1 = np.random.normal(0, 1, nobs)
    S = np.random.normal(0, 1, nobs)

    # Calculate x2, x3, y, and x4 based on the given relationships
    x2 = 3 * x1 + delta
    x3 = x2 + S
    y = S + epsilon
    x4 = y + gamma

    # Create a DataFrame with the generated data
    data = {
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'y': y
    }

    return pd.DataFrame(data)


def generate_rvq_data(nobs):
    """
    Generates data for the 'Redundant Variables Question' scenario.

    :param nobs: Number of observations. Default is 1000.
    :return: A DataFrame containing the generated data.
    """
    # Generate F0 and F1 from a Bernoulli distribution with p=0.5
    F0 = np.random.binomial(1, 0.5, nobs)
    F1 = np.random.binomial(1, 0.5, nobs)

    # F2 is fully redundant with F1
    F2 = F1

    # Calculate y based on the given relationship
    y = F0 + 2 * F1

    # Create a DataFrame with the generated data
    data = {
        'F0': F0,
        'F1': F1,
        'F2': F2,
        'y': y
    }

    return pd.DataFrame(data)


def generate_svq_data(nobs):
    """
    Generates data for the 'Synergistic Variables Question' (SVQ) scenario.

    :param nobs: Number of observations. Default is 1000.
    :return: A DataFrame containing the generated data.
    """
    # Generate F0 and F1 from a Bernoulli distribution with p=0.5
    F0 = np.random.binomial(1, 0.5, nobs)
    F1 = np.random.binomial(1, 0.5, nobs)

    # Calculate Y as XOR(F0, F1)
    y = np.logical_xor(F0, F1).astype(int)

    # Create a DataFrame with the generated data
    data = {
        'F0': F0,
        'F1': F1,
        'y': y
    }

    return pd.DataFrame(data)


def generate_msq_data(nobs):
    """
    Generates data for the 'Multiple Subsets Question' (MSQ) scenario.

    :param nobs: Number of observations. Default is 1000.
    :return: A DataFrame containing the generated data.
    """
    # Generate F1 and F2 from a Bernoulli distribution with p=0.5
    F1 = np.random.binomial(1, 0.5, nobs)
    F2 = np.random.binomial(1, 0.5, nobs)

    # Calculate F0 and Y based on the given relationship
    F0 = F1 + F2
    y = F0

    # Create a DataFrame with the generated data
    data = {
        'F0': F0,
        'F1': F1,
        'F2': F2,
        'y': y
    }

    return pd.DataFrame(data)
