import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from itertools import chain, combinations

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


