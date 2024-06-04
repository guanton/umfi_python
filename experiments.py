import pandas as pd
import numpy as np
from UMFI import UMFI
from MCI import MCI


def getResultsPlot(simTest, nobs, niter, nX):
    Imp = {
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
            df = pd.DataFrame({'x1': A + B, 'x2': B + C, 'x3': D + E, 'x4': E + G})
            df['y'] = df['x1'] + df['x2'] + np.sign(df['x1'] * df['x2']) + df['x3'] + df['x4']

        # Calculate feature importance
        Imp["UMFI_LR"][i, :] = UMFI(df.iloc[:, :-1], df['y'], preprocessing="lr")
        Imp["UMFI_OT"][i, :] = UMFI(df.iloc[:, :-1], df['y'], preprocessing="ot")

    # Normalize feature importance
    Imp2 = Imp.copy()
    for key, value in Imp.items():
        Imp2[key] = (value * 100) / value.sum(axis=1).reshape(-1, 1)

    # Convert results to data frames and concatenate properly
    results = []
    for key, value in Imp2.items():
        df_result = pd.DataFrame(value, columns=df.columns[:-1])
        df_result['method'] = key
        results.append(df_result)

    final_df = pd.concat(results, ignore_index=True)

    return final_df


