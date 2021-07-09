import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
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


class LabelEncoderNew(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.tests = {}
        for col in X.columns:
            self.tests[col] = X[col][1]
        
        return self
    
    def transform(self, X, y=None):
        
        X = X.copy()
        for k, v in self.tests.items():
            X[k] = (X[k] == v).astype(int)
    
        return X


# Finished works good
class City(BaseEstimator, TransformerMixin):
    # # City (feature)
    # Splitting data into categories
    def fit(self, X, y=None):
        city_cut = pd.cut(X.City.value_counts(),
                          right=True,
                          bins=[0, 100, 1000, 5000, 99999],
                          labels=['S', 'M', 'B', 'L']
                          )
        
        self.replace_data_ = city_cut.to_dict()
        
        mask = city_cut.notna()
        notna = city_cut[mask]
        self.most_frequent_ = notna.mode()[0]  # returns most frequent element
        
        return self
    
    def transform(self, X, y=None):
        # splits cities into groups depending on the number of citizens
        # replaces NaN values with random not NaN values in City feature
        X = X.copy()
        X.loc[:, 'City'] = (X['City'].map(self.replace_data_)
                            .fillna(self.most_frequent_)
                            )
        return X


class Source(BaseEstimator, TransformerMixin):
    # # Source (feature)
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        mask = ~X['Source'].isin({'S122', 'S133'})
        X.loc[mask, 'Source'] = 'Other'
        
        return X


class Income(BaseEstimator, TransformerMixin):
    #
    # # Monthly income (feature)
    #
    # # checking outliers for this feature
    # px.line(data.Monthly_Income.quantile(np.arange(0,1,0.01))).show('browser')  # based on plot we take 95 percentile
    # # changing outliers to the value of 95th percentile

    def fit(self, X, y=None):
        self._monthly_income = X.Monthly_Income.quantile(0.95)
    
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        mask = X.Monthly_Income > X.Monthly_Income.quantile(0.95)
        X.Monthly_Income[mask] = self._monthly_income
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
