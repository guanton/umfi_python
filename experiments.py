import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from joblib import Parallel, delayed
from itertools import chain, combinations
import random
import pandas as pd


# Define preprocess_ot, preprocess_lr, and other functions as per the previous translations

# Feature importance functions
def MCI(X, y, k):
    colvec = list(range(X.shape[1]))
    CompleteSet = list(chain.from_iterable(combinations(colvec, r) for r in range(1, k + 1)))
    CompleteSetErrors = [0] * len(CompleteSet)

    for e in range(len(CompleteSetErrors)):
        if len(CompleteSet[e]) > 0:
            rfmod = RandomForestRegressor(n_estimators=100)
            rfmod.fit(X.iloc[:, list(CompleteSet[e])], y)
            if np.issubdtype(y.dtype, np.number):
                CompleteSetErrors[e] = max(r2_score(y, rfmod.predict(X.iloc[:, list(CompleteSet[e])])), 0)
            else:
                CompleteSetErrors[e] = max(1 - accuracy_score(y, rfmod.predict(X.iloc[:, list(CompleteSet[e])])), 0.5)

                if np.issubdtype(y.dtype, np.number):
                    CompleteSetErrors = [max(error, 0) for error in CompleteSetErrors]
                else:
                    CompleteSetErrors = [max(error, 0.5) for error in CompleteSetErrors]

                OUTPUT = [0] * X.shape[1]
                for j in range(X.shape[1]):
                    jsHERE = [list(j in subset) for subset in CompleteSet]
                jSET = [CompleteSet[i] for i, is_here in enumerate(jsHERE) if is_here]

                NOjSET = [list(set(subset) - {j}) for subset in jSET]
                NOjSET = list(set(NOjSET).intersection(CompleteSet))
                jSET = [list(set(subset).union({j})) for subset in NOjSET]
                jSET = [list(sorted(subset)) for subset in jSET]

                charlistjSET = ["".join(map(str, subset)) for subset in jSET]
                charlistNOjSET = ["".join(map(str, subset)) for subset in NOjSET]
                charlistCompleteSet = ["".join(map(str, subset)) for subset in CompleteSet]
                errorWITH = [CompleteSetErrors[charlistCompleteSet.index(charlist)] for charlist in charlistjSET]
                errorWITHOUT = [CompleteSetErrors[charlistCompleteSet.index(charlist)] for charlist in charlistNOjSET]

                OUTPUT[j] = max([errorWITH[i] - errorWITHOUT[i] for i in range(len(errorWITH))])
    return OUTPUT


def MCI_par(X, y, k):
    colvec = list(range(X.shape[1]))
    CompleteSet = list(chain.from_iterable(combinations(colvec, r)) for r in range(1, k + 1))
    CompleteSetErrors = []

    for e in range(len(CompleteSet)):
        if len(CompleteSet[e]) > 0:
            rfmod = RandomForestRegressor(n_estimators=100)
            rfmod.fit(X.iloc[:, list(CompleteSet[e])], y)
    if np.issubdtype(y.dtype, np.number):
        CompleteSetErrors.append(max(r2_score(y, rfmod.predict(X.iloc[:, list(CompleteSet[e])])), 0))
    else:
        CompleteSetErrors.append(max(1 - accuracy_score(y, rfmod.predict(X.iloc[:, list(CompleteSet[e])])), 0.5))

    CompleteSetErrors = [0] + CompleteSetErrors  # Add accuracy for no features

    if np.issubdtype(y.dtype, np.number):
        CompleteSetErrors = [max(error, 0) for error in CompleteSetErrors]
    else:
        CompleteSetErrors = [max(error, 0.5) for error in CompleteSetErrors]

    OUTPUT = [0] * X.shape[1]
    for j in range(X.shape[1]):
        jsHERE = [list(j in subset) for subset in CompleteSet]
    jSET = [CompleteSet[i] for i, is_here in enumerate(jsHERE) if is_here]

    NOjSET = [list(set(subset) - {j}) for subset in jSET]
    NOjSET = list(set(NOjSET).intersection(CompleteSet))
    jSET = [list(set(subset).union({j})) for subset in NOjSET]
    jSET = [list(sorted(subset)) for subset in jSET]

    charlistjSET = ["".join(map(str, subset)) for subset in jSET]
    charlistNOjSET = ["".join(map(str, subset)) for subset in NOjSET]
    charlistCompleteSet = ["".join(map(str, subset)) for subset in CompleteSet]
    errorWITH = [CompleteSetErrors[charlistCompleteSet.index(charlist)] for charlist in charlistjSET]
    errorWITHOUT = [CompleteSetErrors[charlistCompleteSet.index(charlist)] for charlist in charlistNOjSET]

    OUTPUT[j] = max([errorWITH[i] - errorWITHOUT[i] for i in range(len(errorWITH))])
    return OUTPUT


# Define getResultsPlot as per the R code

def getResultsPlot(simTest, nobs, niter, nX):
    Imp = {
        "MCI": np.zeros((niter, nX)),
        "UMFI_LR": np.zeros((niter, nX)),
        "UMFI_OT": np.zeros((niter, nX))
    }

    for i in range(niter):
        if simTest == "Correlated_Interaction":
            A = np.random.normal(0, 1, nobs)
            B = np.random.normal(0, 1, nobs)
            C = np.random.normal(0, 1, nobs)
            D = np.random.normal(0, 1, nobs)
            E = np.random.normal(0, 1, nobs)
            G = np.random.normal(0, 1, nobs)
            Boston = pd.DataFrame({'x1': A + B, 'x2': B + C, 'x3': D + E, 'x4': E + G})
            Boston['y'] = Boston['x1'] + Boston['x2'] + np.sign(Boston['x1'] * Boston['x2']) + Boston['x3'] + Boston[
                'x4']

        # Continue defining scenarios and data generation

        Imp["MCI"][i, :] = MCI(Boston.iloc[:, :-1], Boston['y'], k=nX)
        Imp["UMFI_LR"][i, :] = UMFI(Boston.iloc[:, :-1], Boston['y'], mod_meth="lin")
        Imp["UMFI_OT"][i, :] = UMFI(Boston.iloc[:, :-1], Boston['y'], mod_meth="otpw")

    Imp2 = Imp.copy()
    for key, value in Imp.items():
        Imp2[key] = (value * 100) / value.sum(axis=1).reshape(-1, 1)

    result_dfs = [pd.DataFrame(data=value, columns=Boston.iloc[:, :-1].columns) for key, value in Imp2.items()]
    df = pd.concat([pd.DataFrame({"name": [key] * niter}) for key in Imp2.keys()], ignore_index=True)
    df = pd.concat([df] + result_dfs, axis=1)

    return df
