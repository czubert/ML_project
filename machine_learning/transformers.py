import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tests = {}
    
    def fit(self, X, y=None):
        for col in X.columns:
            self.tests[col] = X[col][1]
        
        return self
    
    def transform(self, X, y=None):
        
        bins = X.copy()
        for k, v in self.tests.items():
            bins[k] = (bins[k] == v).astype(int)

        return bins


class DobToAge(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Create age variable:
        age = pd.DataFrame()
        age['Age'] = X['DOB'].apply(lambda x: int(x[-2:])) - X['Lead_Creation_Date'].apply(lambda x: int(x[-2:]))
        return age

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


class ValueGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, limit):
        self.limit = limit
    
    def fit(self, X, y=None):
        self.feature_name = X.columns[0]
        counts = X.iloc[:, 0].value_counts(dropna=False)
        self.rare_counts = list(counts[counts < self.limit].index)
        
        return self

    def transform(self, X, y=None):
        grouped_values = X.copy()
        mask = grouped_values[self.feature_name].isin(self.rare_counts)
        grouped_values.loc[mask, self.feature_name] = "Other"
        return grouped_values


class Income(BaseEstimator, TransformerMixin):
    """
    changing outliers to the value of 95th percentile Monthly_Income feature
    """
    
    # # checking outliers for this feature
    # px.line(data.Monthly_Income.quantile(np.arange(0,1,0.01))).show('browser')  # based on plot we take 95 percentile
    
    def fit(self, X, y=None):
        self._monthly_income = X.Monthly_Income.quantile(0.95)
        return self

    def transform(self, X, y=None):
        income = X.copy()
        mask = income.Monthly_Income > self._monthly_income
        income.Monthly_Income[mask] = self._monthly_income
    
        return income
