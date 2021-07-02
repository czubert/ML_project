#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd

df_fast_results = pd.read_csv('models.csv')
cols = df_fast_results.columns[:-1]
df_fast_results['std'] = round(df_fast_results[cols].std(axis=1), 7)

# In[3]:
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
import classes

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

#
# # Dealing with binary features
#
bin_feature = ['Gender', 'Mobile_Verified', 'Filled_Form', 'Device_Type']

bin_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(bin_feature)),
    ("bin", classes.BinEncoder()),
])
# bin_pipeline.fit_transform(X_train)

#
# # Dealing with numerical features, standarization
#
# WHAT a lot of values to normalize is set to '-1' which gives negative results after normalization
# probably should be changed to mean or mode
features_to_normalize = ['Loan_Amount_Submitted', 'Loan_Tenure_Submitted', 'Loan_Amount_Applied', 'Loan_Tenure_Applied',
                         'Var5', 'EMI_Loan_Submitted', 'Processing_Fee', 'Interest_Rate', 'Monthly_Income']

# TODO check if standard scaler or normalization is better for the data
num_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(features_to_normalize)),
    ("scaler", StandardScaler()),
    ("back_to_df", classes.BackToDf(features_to_normalize))
])
num_pipeline.fit_transform(X_train)

#
# # Pipeline for categorical data. Replaces NaN with the most frequent value. Then OneHotEncoding is dividing data
#
to_one_hot = ['Var1', 'Var2', 'Var4']

cat_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(to_one_hot)),
    ("impute", classes.MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False, handle_unknown='ignore')),
])
# cat_pipeline.fit_transform(X_train)

city_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(['City'])),
    ("city", classes.City()),
    ("impute", classes.MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False, handle_unknown='ignore')),
])
# city_pipeline.fit_transform(X_train)

source_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(['Source'])),
    ("source", classes.Source()),
    ("impute", classes.MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False, handle_unknown='ignore')),
])
# source_pipeline.fit_transform(X_train)

income_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(['Monthly_Income'])),
    ("income", classes.Income()),
])
# income_pipeline.fit_transform(X_train)


feature_pipeline = FeatureUnion(transformer_list=[
    ("bin_pipeline", bin_pipeline),
    ("city_pipeline", city_pipeline),
    ("source_pipeline", source_pipeline),
    ("income_pipeline", income_pipeline),
    ("cat_pipeline", cat_pipeline),
    ("num_pipeline", num_pipeline),
])

preprocess_pipeline = Pipeline([
    ("preprocess_pipe", feature_pipeline),
    ("fillna_new", SimpleImputer(strategy='constant', fill_value=-1))
])
# X_train_prep_filled = preprocess_pipeline.fit_transform(X_train)


# #
# # # Machine Learning Part
# #

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# # CV
seed = 123
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

# # DecisionTreeClassifier
from imblearn.pipeline import Pipeline

pipe = Pipeline([
    ('preprocessing', preprocess_pipeline),
    #     ('imba', ADASYN()),
    ('classifier', DecisionTreeClassifier()),
])

param_grid = {
    'classifier__max_features': [1, 5, 10]
}

grid_1 = GridSearchCV(pipe, param_grid, cv=kfold)
grid_1.fit(X_train, y_train)
print(grid_1.best_params_)

# #
# # # RandomForestClassifier
# #
# pipe = Pipeline([
#     ('preprocessing', preprocess_pipeline),     
#     ('classifier', RandomForestClassifier())])


# param_grid = {
#             'classifier__max_features': [1, 5, 10]
# }

# grid_2 = GridSearchCV(pipe, param_grid, cv=kfold)
# grid_2.fit(X_train, y_train)
# grid_2.best_params_


# In[6]:


# from imblearn.metrics import classification_report_imbalanced

# Show the classification report
# print(classification_report_imbalanced( y_test, grid_1.best_estimator_.predict(X_test) ))
# print(classification_report_imbalanced( y_test, grid_2.best_estimator_.predict(X_test) ))


# In[ ]:
