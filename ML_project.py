import warnings

import pandas as pd
#
# # warnings
#
import shutup
from joblib import dump
from sklearn.model_selection import train_test_split

from machine_learning import pipelines, estimators

SCORES = 'downloads/scores.csv'

shutup.please()
warnings.filterwarnings("ignore")

#
# # Data for classification
#
data = pd.read_csv('data/Train_nyOWmfK.csv', encoding="latin1")
test = pd.read_csv('data/Test_bCtAN1w.csv', encoding="latin1")

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
# # Train, Test, Validation Split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

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
    'X_val': X_val,
    'X_test': X_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
}

scores, models = estimators.get_best_classsifier(**ml_variables)

#
# # Saving scores to file
#
try:
    saved_scores = pd.read_csv(SCORES)
    new_scores = pd.DataFrame(scores)
    scores = pd.concat([saved_scores, new_scores], axis=1)
    pd.DataFrame(scores).to_csv(SCORES)
except FileNotFoundError:
    pd.DataFrame(scores).to_csv(SCORES)
#
# # Saving models to files
#
for model, values in models.items():
    dump(values, f'models/{"_".join(model.split())}_model.joblib')
