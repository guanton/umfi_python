from umfi.preprocess_OT import *
import pyreadr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

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
    y = F0 + 10 * F1

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

def generate_sg_data(nobs):
    """
    Generates data for the 'Synthetic genes' (SG) dataset.

    :param nobs: Number of observations.
    :return: A DataFrame containing the generated data.
    """
    # Generate Y from a Bernoulli distribution with p=0.5
    y = np.random.binomial(1, 0.5, nobs)

    f1 = np.zeros(nobs)
    f2 = np.zeros(nobs)
    f3 = np.zeros(nobs)

    for i in range(nobs):
        if y[i] == 0:
            # For healthy samples (y=0)
            prob_f1_f2 = np.random.rand()
            if prob_f1_f2 < 0.95:
                f1[i] = 1
                f2[i] = 1
            else:
                f1_f2_joint_state = np.random.choice([0, 1], size=2, replace=True)
                f1[i] = f1_f2_joint_state[0]
                f2[i] = f1_f2_joint_state[1]
            f3[i] = np.random.binomial(1, 0.2)
        else:
            # For cancerous samples (y=1)
            prob_f1_f2 = np.random.rand()
            if prob_f1_f2 < 0.05:
                f1[i] = 1
                f2[i] = 1
            else:
                f1_f2_joint_state = np.random.choice([0, 1], size=2, replace=True)
                f1[i] = f1_f2_joint_state[0]
                f2[i] = f1_f2_joint_state[1]
            f3[i] = np.random.binomial(1, 0.8)

    # Create a DataFrame with the generated data
    data = {
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'y': y
    }


    return pd.DataFrame(data)




def generate_terc1_data(nobs):
    """
    Generates data for the 'TERC-1' scenario.

    :param nobs: Number of observations.
    :return: A DataFrame containing the generated data.
    """
    # Generate F0, F1, F2 from a Bernoulli distribution with p=0.5
    F0 = np.random.binomial(1, 0.5, nobs)
    F1 = np.random.binomial(1, 0.5, nobs)
    F2 = np.random.binomial(1, 0.5, nobs)

    # F3, F4, F5 are copies of F0
    F3 = F0
    F4 = F0
    F5 = F0

    # Calculate Y based on the condition
    y = np.where((F0 == F1) & (F1 == F2), 0, 1)

    # Create a DataFrame with the generated data
    data = {
        'F0': F0,
        'F1': F1,
        'F2': F2,
        'F3': F3,
        'F4': F4,
        'F5': F5,
        'y': y
    }

    return pd.DataFrame(data)


def generate_terc2_data(nobs):
    """
    Generates data for the 'TERC-2' scenario.

    :param nobs: Number of observations.
    :return: A DataFrame containing the generated data.
    """

    # Generate F0, F1, F2 from a Bernoulli distribution
    F0 = np.random.binomial(1, 0.5, nobs)  # np.random.normal(0, 1, nobs)
    F1 = np.random.binomial(1, 0.5, nobs)
    F2 = np.random.binomial(1, 0.5, nobs)

    # F3 is a copy of F0, F4 is a copy of F1, F5 is a copy of F2
    F3 = F0
    F4 = F1
    F5 = F2

    # Calculate Y based on the condition
    y = np.where((F0 == F1) & (F1 == F2), 0, 1)

    # Create a DataFrame with the generated data
    data = {
        'F0': F0,
        'F1': F1,
        'F2': F2,
        'F3': F3,
        'F4': F4,
        'F5': F5,
        'y': y
    }

    return pd.DataFrame(data)


def generate_mixed_cts_discrete_data(nobs):
    x0 = np.random.normal(0, 1, nobs)
    x1 = np.random.uniform(-0.5, 0.5, nobs)

    # Generate variables x1 and S from a standard normal distribution N(0, 1)
    x2 = np.random.normal(0, 1, nobs)
    x3 = np.random.binomial(4, 0.5, nobs)- 2
    x4 = np.random.binomial(4, 0.5, nobs)
    # # Scale each variable to have variance 1
    # x0 /= np.std(x0)
    # x1 /= np.std(x1)
    # x2 /= np.std(x2)
    # x3 /= np.std(x3)
    # x4 /= np.std(x4)

    # Calculate Y based on the condition
    y = 10*np.sign(x0*x3) + 2*x2 + x4

    # Create a DataFrame with the generated data
    data = {
        'x0': x0,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'y': y
    }
    return pd.DataFrame(data)

def obtain_BRCA_data():
    BRCA_data = pyreadr.read_r('data/BRCA.rda')
    BRCA_df = list(BRCA_data.values())[0]
    BRCA_df = BRCA_df.drop(columns='Sample.ID')
    return BRCA_df





