# # #
# # # # Machine Learning Part
# # #

import pandas as pd
# # imblearn
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
# # saving models
from joblib import dump
# # feature selection
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
# # metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# # classifiers

# # CV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SCORE_PATH = 'scores.csv'
SEED = 123
classifiers = {
    # # Done
    # 'LogisticRegression':
    #     {
    #         'name': 'LogisticRegression',
    #         'estimator': LogisticRegression(),
    #         'params':
    #             {
    #                 "classifier__class_weight": ['balanced'],
    #                 "classifier__C": [1],
    #                 "classifier__max_iter": [90],
    #                 "classifier__solver": ['saga'],
    #                 "classifier__penalty": ['l1'],
    #                 "classifier__tol": [0.0001],
    #                 'selector__k': [100],
    #                 'decomposition__n_components': [100],
    #             }},
    
    # # Done
    # 'SVC':
    #     {
    #         'name': 'SVC',
    #         'estimator': SVC(),
    #         'params':
    #             {
    #                 "classifier__C": [2],
    #                 "classifier__kernel": ['rbf'],
    #                 "classifier__degree": [1],
    #                 "classifier__max_iter": [-1],
    #                 "classifier__gamma": ['auto'],
    #                 "classifier__tol": [0.1],
    #                 "classifier__probability": [True],
    #                 'selector__k': [150],
    #             }},
    
    # # Done
    # 'AdaBoostClassifier':
    #     {
    #         'name': 'AdaBoostClassifier',
    #         'estimator': AdaBoostClassifier(),
    #         'params':
    #             {
    #                 'classifier__n_estimators': [500],
    #                 'classifier__learning_rate': [0.1],
    #             }},
    
    # # Done
    # 'DecisionTreeClassifier':
    #     {
    #         'name': 'DecisionTreeClassifier',
    #         'estimator': DecisionTreeClassifier(),
    #         'params':
    #             {
    #                 'classifier__max_features': [0.5],
    #                 'classifier__max_depth': [10],
    #                 'classifier__criterion': ['gini'],
    #                 'classifier__max_leaf_nodes': [10],
    #                 'classifier__min_weight_fraction_leaf': [0.01],
    #                 'classifier__min_samples_split': [5],
    #                 'selector__k': [100],
    #                 'decomposition__n_components': [90],
    #             }},
    
    # # Done
    # 'XGBoostClassifier':
    #     {
    #         'name': 'XGBoostClassifier',
    #         'estimator': XGBClassifier(),
    #         'params':
    #             {
    #                 'classifier__n_estimators': [1000],
    #                 'classifier__learning_rate': [0.01],
    #                 'classifier__max_depth': [4],
    #                 'classifier__objective': ['binary:logistic'],
    #                 'classifier__booster': ['gbtree'],
    #                 'classifier__tree_method': ['auto'],
    #                 'classifier__gamma': [0],
    #                 'classifier__min_child_weight': [6],
    #                 'classifier__reg_alpha': [0.005],
    #                 'classifier__reg_lambda': [0.01],
    #                 'classifier__subsample': [0.9],
    #                 'classifier__colsample_bytree': [0.8],
    #                 'selector__k': [150],
    #             }},
    
    'RandomForestClassifier':
        {
            'name': 'RandomForestClassifier',
            'estimator': RandomForestClassifier(),
            'params':
                {
                    'classifier__n_estimators': [1000],
                    'classifier__criterion': ['gini'],
                    'classifier__max_features': [0.5],
                    'classifier__max_depth': [10],
                    'classifier__max_leaf_nodes': [150],
                    'classifier__min_samples_split': [5],
                    'classifier__bootstrap': [True, False],
                    'classifier__max_samples': [150],
                    'selector__k': [150],
                }},
    
    # 'ExtraTreesClassifier':
    #     {
    #         'name': 'ExtraTreesClassifier',
    #         'estimator': ExtraTreesClassifier(),
    #         'params':
    #             {
    #                 'classifier__n_estimators': [100,200],
    #                 'classifier__criterion': ['gini', 'entropy'],
    #                 'classifier__max_depth': [10,50,100],
    #                 'classifier__max_features': [1,10,50,100],
    #                 'classifier__min_samples_split': [1,2,5],
    #                 'classifier__min_samples_leaf': [1,2,5],
    #                 'classifier__max_samples': [1,10,100],
    #                 'selector__k': [50,100,150],
    #             }},
    
}

kfold = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)


def get_best_classsifier(preprocess_pipeline, X_train, X_val, X_test, y_train, y_test, y_val):
    """
    Getting the best models for all passed estimators and saves it to joblib file.
    Using GridSearchCV for the getting the best params. Saving scores to csv file
    :param preprocess_pipeline: Pipeline that preprocess the data
    :param X_train: DataFrame of data for training
    :param X_val:  DataFrame of data for validation
    :param X_test: DataFrame of data for testing
    :param y_train: DataFrame of labels for training
    :param y_val: DataFrame of labels for validation
    :param y_test: DataFrame of labels for testing
    :return: DataFrame of obtained scores for each estimator, Dict of trained models
    """
    
    scores_index = ['best_params', 'best_score', 'roc_auc_score_train', 'roc_auc_score_val', 'roc_auc_score_test']
    scores = pd.DataFrame(index=scores_index)
    models = {}
    
    for key, value in classifiers.items():
        tmp_pipe = Pipeline([
            ('preprocessing', preprocess_pipeline),
            ('scaler', StandardScaler(with_mean=False)),
            ('sampling', RandomUnderSampler(random_state=40)),
            ('selector', SelectKBest()),
            # ('decomposition', PCA()),
            ('classifier', value['estimator'])
        ])

        # WHAT: da się do grida wrzucić jakoś różne parametry dla metod z tmp_pipe?
        grid = GridSearchCV(tmp_pipe, value['params'], cv=kfold, scoring='roc_auc')
        grid.fit(X_train, y_train)

        roc_auc_score_train = roc_auc_score(y_train, grid.predict_proba(X_train)[:, 1])
        roc_auc_score_val = roc_auc_score(y_val, grid.predict_proba(X_val)[:, 1])
        roc_auc_score_test = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])

        # Adding params and scores of a model to DataFrame
        scores[key] = (grid.best_params_, grid.best_score_, roc_auc_score_train, roc_auc_score_val, roc_auc_score_test)

        # storing best model in the DataFrame
        models[key] = grid.best_estimator_
        
        #
        # # Saving models to files
        #
        dump(grid.best_estimator_, f'models/{key}_model.joblib')
        
        print(f'{key} has been processed')
    #
    # # Saving scores to file
    #
    try:
        saved_scores = pd.read_csv(SCORE_PATH)
        saved_scores.index = scores_index
        scores = pd.concat([saved_scores, scores], axis=1)
        scores.index = scores_index
        scores.to_csv(SCORE_PATH, index=False)
    except FileNotFoundError:
        scores.index = scores_index
        scores.to_csv(SCORE_PATH, index=True)
    
    return scores, models
