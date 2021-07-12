import pandas as pd
from sklearn.model_selection import train_test_split

import preprocessing
import estimators

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

#
# # Machine Learning
#
ml_variables = {
    'preprocess_pipeline': preprocess_pipeline,
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'k_best': 50,
}

report, models = estimators.get_best_classsifier(**ml_variables)
