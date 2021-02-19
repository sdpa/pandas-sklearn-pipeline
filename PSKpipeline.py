from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd

class ColumnExtractor(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xcols = X[self.cols]
        return Xcols

class NanToNONETransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.fillna('NONE')
        return X

class DFNumImputer(TransformerMixin):
    def __init__(self, missing = None, strategy = 'mean', fillValue = None):
        self.strategy = strategy
        self.imp = None
        self.fillValue = fillValue,
        self.missing = missing,
    def fit(self, X, y=None):
        self.imp = SimpleImputer(strategy = self.strategy, missing_values = self.missing)
        self.imp.fit(X)
        return self
    def transform(self, X):
        result = self.imp.transform(X)
        resultDF = pd.DataFrame(result, index = X.index, columns = X.columns)
        return resultDF

class CatImputer(TransformerMixin):
    def __init__(self, missing = None, strategy = None, fillValue = None):
        self.strategy = strategy
        self.fillValue = fillValue,
        self.missing = missing,
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.strategy == 'most-frequent':
            for column in X:
                X[column].fillna(X[column].value_counts().idxmax())
        elif self.strategy == 'constant':
            try:
                if(self.fillValue == None):
                    raise ValueError('fillValue must be provided with a valid value')
            except ValueError as e:
                print(e)
            else:
                X.fillna(self.fillValue)
        return X

#Repalce the 'fill' value with 'source' if 'fill' is Null. 
class FillNAWithOther(TransformerMixin):
    def __init__(self, fill, source):
        self.fill = fill
        self.source = source
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X[self.fill] = np.where(X[self.fill].isnull(), X[self.source], X[self.fill])
        return X