import pandas as pd
from sklearn.preprocessing import StandardScaler


class Scaler(StandardScaler):
    '''
    Specific for dealing with pd.DataFrame.
    Don't scale the dummy features.
    '''

    def fit(self, Xtrain):
        self.__num_cols = Xtrain.columns[Xtrain.dtypes != bool]
        self.__dummy_cols = Xtrain.columns[Xtrain.dtypes == bool]
        Xnumerical = Xtrain[self.__num_cols]
        super(Scaler, self).fit(Xnumerical)
        return self

    def transform(self, X):
        Xnumerical = X[self.__num_cols]
        Xdummy = X[self.__dummy_cols]
        scaledXnumerical = super(Scaler, self).transform(Xnumerical)
        Xnumerical = pd.DataFrame(scaledXnumerical, index=Xnumerical.index, columns=Xnumerical.columns)
        return pd.concat([Xnumerical, Xdummy], axis=1)
