# # #
# # # # Machine Learning Part
# # #

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

# classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

# imblearn
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

# # CV
seed = 123
classifiers = {
    # 'RF':
    #     {
    #         'name': 'Random Forest Classifier',
    #         'estimator': RandomForestClassifier(),
    #         'params':
    #             {
    #                 'classifier__n_estimators': [50, 100, 150, 250, 500],  # sprobowac jeszcze jakies dodac
    #                 'classifier__criterion': ['gini','entropy'],
    #                 'classifier__max_features': [0.3,0.4,0.5],
    #                 'classifier__max_depth': [5, 10, 16],
    #                 'classifier__max_leaf_nodes': [10, 30, 50, 100],
    #                 'classifier__min_samples_split': [1, 2, 3],
    #                 'classifier__bootstrap': [True, False],
    #                 'classifier__max_samples': [1, 10, 100],
    #                 'selector__k': [None,100,150,200],
    #             }},
    
    # 'tree':
    #     {
    #         'name': 'Decision Tree Classifier',
    #         'estimator': DecisionTreeClassifier(),
    #         'params':
    #             {
    #                 'classifier__max_features': [0.1,0.3,0.5],
    #                 'classifier__max_depth': [5,6,7,10,15],
    #                 'classifier__criterion': ['gini'],
    #                 'classifier__max_leaf_nodes': [50, 100],
    #                 'classifier__min_weight_fraction_leaf': [0, 1,10,100],
    #                 'classifier__min_samples_split': [0.1, 1,2,10,100],
    #
    #                 'selector__k': [40],
    #             }},
    
    'Logistiic':
        {
            'name': 'Logistic regression',
            'estimator': LogisticRegression(),
            'params':
                {
                    # "classifier__class_weight": [None, 'balanced'],
                    # "classifier__C": [0.1,1,10],
                    # "classifier__max_iter": [100],
                    # "classifier__penalty": ['l1', 'l2', 'elasticnet', 'none'], #potem sprawdzić z tym, ale bez solvera
                    # "classifier__solver": ['saga'],
                    "classifier__solver": ['lbfgs'],
                    # "classifier__tol": [0.000001,0.00001],
                    'selector__k': [100],
                }},
    
    #     'XGB':
    #         {
    #             'name': 'XGBoost Classifier',
    #             'estimator': XGBClassifier(),
    #             'params':
    #                 {
    #                     'classifier__n_estimators': [100,200],
    #                     'classifier__max_depth': [10,50,100],
    #                     'classifier__gamma':[1],
    #                     'classifier__reg_alpha': [0],
    #                     'classifier__reg_lambda': [0.2],
    #                 }},
    #
    # 'SVC':
    #     {
    #         'name': 'SVC classifier',
    #         'estimator': SVC(),
    #         'params':
    #             {
    #                 # "classifier__kernel": ["poly"],
    #                 "classifier__degree": [1, 2, 3],
    #                 "classifier__C": [0.1, 1, 10],
    #                 'selector__k': [40,50,60],
    #             }},
}
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)


def get_best_classsifier(preprocess_pipeline, X_train, y_train, X_test, y_test):
    # TODO make preprocessing adasyn etc before the loop, not to repeat this fo each estimators
    # final_processing_pipe = Pipeline([
    #     ('preprocessing', preprocess_pipeline),
    #     ('sampling', ADASYN(random_state=55)),
    #     ('selector', SelectKBest(k=k_best)),
    #     ('decomposition', PCA()),
    # ])
    
    scores = {}
    models = {}
    
    for key, value in classifiers.items():
        tmp_pipe = Pipeline([
            # ('final_processing', final_processing_pipe),
            ('preprocessing', preprocess_pipeline),
            # ('sampling', ADASYN(random_state=k_best)),
            ('sampling', RandomUnderSampler(random_state=40)),
            ('selector', SelectKBest()),
            # ('decomposition', PCA(20)),
            ('classifier', value['estimator'])
        ])
        # WHAT: da się do grida wrzucić jakoś różne parametry dla metod z tmp_pipe?
        grid = GridSearchCV(tmp_pipe, value['params'], cv=kfold, scoring='roc_auc')
        grid.fit(X_train, y_train)

        roc_auc_score_train = roc_auc_score(y_train, grid.predict_proba(X_train)[:, 1])
        roc_auc_score_test = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])

        scores[value['name']] = {
            'best_params': grid.best_params_,
            'roc_auc_score_train_test': (roc_auc_score_train, roc_auc_score_test)
        }

        models[value['name']] = grid.best_estimator_
        print(f'{key} has been processed')
    return scores, models
