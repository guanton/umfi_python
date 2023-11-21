import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from joblib import Parallel, delayed
from preprocess_LR import preprocess_lr
from preprocess_OT import preprocess_ot


def umfi(X, y, mod_meth="lr"):
    fi = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        if mod_meth == "ot":
            newX = preprocess_ot(X, i)
        elif mod_meth == "lr":
            newX = preprocess_lr(X, i)

        if np.issubdtype(y.dtype, np.number):
            rf_with = RandomForestRegressor(n_estimators=100)
            rf_with.fit(newX, y)
            r2_with = max(r2_score(y, rf_with.predict(newX)), 0)

            newX_without = newX.drop(columns=[i])
            rf_without = RandomForestRegressor(n_estimators=100)
            rf_without.fit(newX_without, y)
            r2_without = max(r2_score(y, rf_without.predict(newX_without)), 0)

            fi[i] = r2_with - r2_without
        elif np.issubdtype(y.dtype, np.object):
            rf_with = RandomForestClassifier(n_estimators=100)
            rf_with.fit(newX, y)
            accuracy_with = max(accuracy_score(y, rf_with.predict(newX)), 0.5)

            newX_without = newX.drop(columns=[i])
            rf_without = RandomForestClassifier(n_estimators=100)
            rf_without.fit(newX_without, y)
            accuracy_without = max(accuracy_score(y, rf_without.predict(newX_without)), 0.5)

            fi[i] = accuracy_with - accuracy_without

    fi[fi < 0] = 0
    return fi


def umfi_par(X, y, mod_meth):
    def calculate_umfi(i):
        if mod_meth == "ot":
            newX = preprocess_ot(X, i)
        elif mod_meth == "lr":
            newX = preprocess_lr(X, i)

        if np.issubdtype(y.dtype, np.number):
            rf_with = RandomForestRegressor(n_estimators=100)
            rf_with.fit(newX, y)
            r2_with = max(r2_score(y, rf_with.predict(newX)), 0)

            newX_without = newX.drop(columns=[i])
            rf_without = RandomForestRegressor(n_estimators=100)
            rf_without.fit(newX_without, y)
            r2_without = max(r2_score(y, rf_without.predict(newX_without)), 0)

            return r2_with - r2_without
        elif np.issubdtype(y.dtype, np.object):
            rf_with = RandomForestClassifier(n_estimators=100)
            rf_with.fit(newX, y)
            accuracy_with = max(accuracy_score(y, rf_with.predict(newX)), 0.5)

            newX_without = newX.drop(columns=[i])
            rf_without = RandomForestClassifier(n_estimators=100)
            rf_without.fit(newX_without, y)
            accuracy_without = max(accuracy_score(y, rf_without.predict(newX_without)), 0.5)

            return accuracy_with - accuracy_without

    fi = Parallel(n_jobs=-1, backend="loky")(delayed(calculate_umfi)(i) for i in range(X.shape[1]))
    fi = np.array(fi)
    fi[fi < 0] = 0
    return fi
