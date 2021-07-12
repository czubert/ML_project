import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder


# A class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.attribute_names]


class MyOneHotEncoder(BaseEstimator, TransformerMixin):
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


class MostFreqImputer(BaseEstimator, TransformerMixin):
    # # City (feature)
    # Splitting data into categories
    def fit(self, X, y=None):
        mask = X.notna()
        notna = X[mask]
        self.most_frequent_ = notna.mode()  # returns most frequent element
        
        return self
    
    def transform(self, X, y=None):
        # splits cities into groups depending on the number of citizens
        # replaces NaN values with random not NaN values in City feature
        X = X.copy()
        X = X.fillna(self.most_frequent_, axis=0)
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
