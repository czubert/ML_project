import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


# A class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.attribute_names]


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


class LabelEncoderNew(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        return X


class BinEncoder():
    def fit(self, X, y=None):
        tests = {}
        for col in X.columns:
            tests[col] = X[col][1]
        
        for k, v in tests.items():
            X[k] = (X[k] == v).astype(int)
        return self
    
    def transform(self, X, y=None):
        
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
        self.most_frequent_ = notna.mode()[0]  # areturns most frequent element
        
        return self
    
    def transform(self, X, y=None):
        # splits cities into groups depending on the number of citizens
        # replaces NaN values with random not NaN values in City feature
        X.loc[:, 'City'] = (X['City'].map(self.replace_data_)
                            .fillna(self.most_frequent_)
                            )
        return X


class Source(BaseEstimator, TransformerMixin):
    # # Source (feature)
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
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
        self.mask_ = X.Monthly_Income > X.Monthly_Income.quantile(0.95)
        self.monthly_income_ = X.Monthly_Income.quantile(0.95)
        
        return self
    
    def transform(self, X, y=None):
        X.Monthly_Income[self.mask_] = self.monthly_income_
        return X


class Fillna_new():
    #
    # # Dealing with nan values
    #
    
    # TODO is it a right idea to get rid of nan?
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.fillna(-1)
        return X
        # return X.fillna(-1, inplace=True)
