import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
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


class EmiLoanSubmitted(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # applies 1 where there is data missing else 0
        X['EMI_Loan_Submitted_Missing'] = X['EMI_Loan_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
        # drop original variables:
        X.drop('EMI_Loan_Submitted', axis=1, inplace=True)
        
        return X


class Submitted(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for col in X.columns:
            # applies 1 where there is data missing else 0
            # X[f'{col}_Missing'] = X[col].apply(lambda x: 1 if pd.isnull(x) else 0)
            X.loc[:, f'{col}_Missing'] = X[col].apply(lambda x: 1 if pd.isnull(x) else 0)
            
            # drop original variables:
            X.drop(col, axis=1, inplace=True)
        
        return X


class DobToAge(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Create age variable:
        X.loc[:, 'Age'] = X['DOB'].apply(lambda x: 115 - int(x[-2:]))
        # drop DOB:
        X.drop('DOB', axis=1, inplace=True)
        
        return X


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
    """
    changing outliers to the value of 95th percentile
    """
    
    # # Monthly income (feature)
    
    # # checking outliers for this feature
    # px.line(data.Monthly_Income.quantile(np.arange(0,1,0.01))).show('browser')  # based on plot we take 95 percentile
    
    def fit(self, X, y=None):
        self._monthly_income = X.Monthly_Income.quantile(0.95)
        
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        mask = X.Monthly_Income > X.Monthly_Income.quantile(0.95)
        X.Monthly_Income[mask] = self._monthly_income
        return X
