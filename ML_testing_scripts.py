import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# A class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet


'''
1. categorial
    a) nominal
    b) ordered
2. Numerical
'''

data = pd.read_csv('data/Train_nyOWmfK.csv', encoding="latin1")
# print(data.shape)
# print(data.info())

#
# # Getting rid of irrelevant features
#
irrelevant_features = ["DOB", "Lead_Creation_Date", "ID", "Employer_Name", "Salary_Account"]
data.drop(irrelevant_features, axis=1, inplace=True)
data.dropna(subset=["Loan_Amount_Applied"], inplace=True)

#
# # getting X and y
#
X = data.drop(["LoggedIn", "Disbursed"], axis=1)
y = data.Disbursed

#
# # Train, Test, Split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.attribute_names]


class BinEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        print(X.shape)
        self.tests = {}
        for col in X.columns:
            self.tests[col] = X[col][1]
        
        return self
    
    def transform(self, X, y=None):
        for k, v in self.tests.items():
            X.loc[:, k] = (X[k] == v).astype(int)
        
        print(X)
        return X


bin_feature = ['Gender', 'Mobile_Verified', 'Filled_Form', 'Device_Type']

bin_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(bin_feature)),
    ("bin", BinEncoder()),
])
asda = bin_pipeline.fit_transform(X_train)

# bins = BinEncoder()
# bins.fit_transform(X_train)