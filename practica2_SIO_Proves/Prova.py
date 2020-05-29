import pandas as pd
import numpy as np
from ast import literal_eval

try:
    from sklearn.datasets import make_regression

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=1000, n_features=6,
                           n_informative=3, n_targets=6,
                           tail_strength=0.5, noise=0.02,
                           shuffle=True, coef=False, random_state=0)

    # Convert to a pandas dataframe like in your example
    icols = ['i0', 'i1', 'i2', 'i3', 'i4', 'i5']
    jcols = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5']
    df = pd.concat([pd.DataFrame(X, columns=icols),
                    pd.DataFrame(y, columns=jcols)], axis=1)

    # Introduce a few np.nans in there
    df.loc[0, jcols] = np.nan
    df.loc[10, jcols] = np.nan
    df.loc[100, jcols] = np.nan

    notnans = df[jcols].notnull().all(axis=1)
    df_notnans = df[notnans]

    print(df_notnans[icols], df_notnans[jcols])
    X_train, X_test, y_train, y_test = train_test_split(df_notnans[icols], df_notnans[jcols],
                                                        train_size=0.75,
                                                        random_state=4)


    regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=30,
                                                              random_state=0))

    # Fit on the train data
    regr_multirf.fit(X_train, y_train)

    # Check the prediction score
    score = regr_multirf.score(X_test, y_test)

    df_nans = df.loc[~notnans].copy()
    print(df_nans)
    df_nans[jcols] = regr_multirf.predict(df_nans[icols])
    df_nans

except Exception as e:
    print(e)