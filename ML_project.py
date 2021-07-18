import pandas as pd
from sklearn.model_selection import train_test_split
from machine_learning import pipelines, estimators, utils, transformers
from joblib import dump, load
#
# # warnings
#
import shutup
import warnings

shutup.please()
warnings.filterwarnings("ignore")

# results obtained by LazyPredict
df_fast_results = pd.read_csv('models.csv')
cols = df_fast_results.columns[:-1]
df_fast_results['std'] = round(df_fast_results[cols].std(axis=1), 7)

#
# # Processing data description
#

"""
City values changed to 'S', 'M', 'B', 'L' depending on the occurrence
DOB converted to Age | DOB dropped

Dropped:
ID dropped - not relevant
Lead_Creation_Date dropped because made little intuitive impact on outcome
LoggedIn, Salary_Account dropped
Loan_Amount_Submitted dropped - highly correlated with EMI_Loan_Submitted (>91%)

Existing_EMI imputed with 0 (median) since only 111 values were missing
Interest_Rate_Missing created which is 1 if Interest_Rate was missing else 0 |
Original variable Interest_Rate dropped
Loan_Amount_Applied, Loan_Tenure_Applied imputed with median values
EMI_Loan_Submitted_Missing created which is 1 if EMI_Loan_Submitted was missing else 0 |
Original variable EMI_Loan_Submitted dropped
Loan_Tenure_Submitted_Missing created which is 1 if Loan_Tenure_Submitted was missing else 0 |
Original variable Loan_Tenure_Submitted dropped
Processing_Fee_Missing created which is 1 if Processing_Fee was missing else 0 |
Original variable Processing_Fee dropped
Source – top 2 kept as is and all others combined into different category
Employer_Name – top n values kept as is and all others combined into different category
Salary_Account – top n values kept as is and all others combined into different category
Numerical and One-Hot-Coding performed
"""

#
# # Data for classification
#
data = pd.read_csv('data/Train_nyOWmfK.csv', encoding="latin1")
# print(data.shape)
# print(data.info())


#
# # Getting rid of irrelevant features
#
irrelevant_features = ['Lead_Creation_Date', 'ID', 'LoggedIn']
data = data.drop(irrelevant_features, axis=1)  # drops features that have no impact on model
data = data.dropna(subset=["Loan_Amount_Applied"])  # drops variables where Loan_Amount_applied is NaN

#
# # getting X and y
#
X = data.drop(['Disbursed'], axis=1)
y = data.Disbursed

#
# # Train, Test, Split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

#
# # Preprocessing data
#
preprocess_pipeline = pipelines.get_preprocessed_data(X_train)
# X_train_prep_filled = preprocess_pipeline.fit_transform(X_train)

#
# # Machine Learning
#
ml_variables = {
    'preprocess_pipeline': preprocess_pipeline,
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
}

scores, models = estimators.get_best_classsifier(**ml_variables)

#
# # Saving scores to file
#
try:
    saved_scores = pd.read_csv('downloads/scores.csv')
    new_scores = pd.DataFrame(scores)
    scores = pd.concat([saved_scores, new_scores], axis=1)
    pd.DataFrame(scores).to_csv('downloads/scores.csv')
except FileNotFoundError:
    pd.DataFrame(scores).to_csv('downloads/scores.csv')
#
# # Saving models to files
#
for model, values in models.items():
    dump(values, f'{model}_model.joblib')
