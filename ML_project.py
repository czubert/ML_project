import pandas as pd
from sklearn.model_selection import train_test_split


# In[1]:
import preprocessing

df_fast_results = pd.read_csv('models.csv')
cols = df_fast_results.columns[:-1]
df_fast_results['std'] = round(df_fast_results[cols].std(axis=1), 7)

# In[3]:


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
data = data.drop(irrelevant_features, axis=1)  # drops features that have no impact on model
data = data.dropna(subset=["Loan_Amount_Applied"])  # drops variables where Loan_Amount_applied is NaN

#
# # getting X and y 
#
X = data.drop(["LoggedIn", "Disbursed"], axis=1)
y = data.Disbursed

#
# # Train, Test, Split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#
# # Preprocessing data
#
preprocess_pipeline = preprocessing.get_preprocessed_data(X_train)
X_train_prep_filled = preprocess_pipeline.fit_transform(X_train)

# # #
# # # # Machine Learning Part
# # #

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# # CV
seed = 123
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

# # DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_selection import SelectKBest

# pipe = Pipeline([
#     ('preprocessing', preprocess_pipeline),
#     ('sampling', SMOTE(random_state=55)),
#     # ('sampling', ADASYN(random_state=55)),
#     ('classifier', DecisionTreeClassifier()),
# ])
#
# param_grid = {
#     'classifier__max_features': [0.25, 0.5],
# }

# grid_1 = GridSearchCV(pipe, param_grid, cv=kfold)
# grid_1.fit(X_train, y_train)
# print(grid_1.best_params_)


#
# # RandomForestClassifier
#
pipe = Pipeline([
    ('preprocessing', preprocess_pipeline),
    ('sampling', ADASYN(random_state=55)),
    ('selector', SelectKBest(k=50)),
    ('classifier', RandomForestClassifier())
])

param_grid = {
    # 'classifier__n_estimators': [10,100,400],
    # 'classifier__criterion' :['gini', 'entropy'],
    # 'classifier__max_features': [0.25,0.5,0.75],
    'classifier__max_depth': [4, 8],
    # 'selector__k' :[40,50,54],
}

grid_2 = GridSearchCV(pipe, param_grid, cv=kfold)
grid_2.fit(X_train, y_train)
print(grid_2.best_params_)

from imblearn.metrics import classification_report_imbalanced

# Show the classification report
# print(classification_report_imbalanced( y_test, grid_1.best_estimator_.predict(X_test) ))
print(classification_report_imbalanced(y_test, grid_2.best_estimator_.predict(X_test)))
