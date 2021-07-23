import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class DataFrameSelector(BaseEstimator, TransformerMixin):
    # A class to select numerical or categorical columns
    # since Scikit-Learn doesn't handle DataFrames yet
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.attribute_names]


class MostFreqImputer(BaseEstimator, TransformerMixin):
    # # Fills NaN with most frequent data
    #
    def fit(self, X, y=None):
        mask = X.notna()
        notna = X[mask]
        self.most_frequent_ = notna.mode()  # returns most frequent element
        
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X = X.fillna(self.most_frequent_, axis=0)
        return X


class MyOneHotEncoder(BaseEstimator, TransformerMixin):
    # classic OneHotEncoder, but returning DataFrame instead of NumPy array
    # It is made to keep the names of the columns
    def __init__(self):
        super().__init__()
        self._encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    def fit(self, X, y=None, cols=None):
        self._encoder.fit(X)
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X = self._encoder.transform(X)
        X = pd.DataFrame(X, columns=self._encoder.get_feature_names())
        return X


class BackToDf(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        super().__init__()
        self._columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self._columns)
        return df


class ToDf(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = pd.DataFrame(X)
        return df


class RemoveOutliers(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.mask = df.quantile(q=np.array([0.05, 0.95]), axis=0)
        return self
    
    def transform(self, X, y=None):
        df = pd.DataFrame(X)
        df = df[(df > self.mask.iloc[0, :]) & (df < self.mask.iloc[1, :])]
        return df
