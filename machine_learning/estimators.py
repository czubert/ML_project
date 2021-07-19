# # #
# # # # Machine Learning Part
# # #

# imblearn
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest
# metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# classifiers
from sklearn.svm import SVC

# # CV
seed = 123
classifiers = {
    # 'RandomForestClassifier':
    #     {
    #         'name': 'RandomForestClassifier',
    #         'estimator': RandomForestClassifier(),
    #         'params':
    #             {
    #                 'classifier__n_estimators': [50, 100, 150, 250, 500],  # sprobowac jeszcze jakies dodac
    #                 # 'classifier__criterion': ['gini','entropy'],
    #                 # 'classifier__max_features': [0.3,0.4,0.5],
    #                 # 'classifier__max_depth': [5, 10, 16],
    #                 # 'classifier__max_leaf_nodes': [10, 30, 50, 100],
    #                 # 'classifier__min_samples_split': [1, 2, 3],
    #                 # 'classifier__bootstrap': [True, False],
    #                 # 'classifier__max_samples': [1, 10, 100],
    #                 # 'selector__k': [None,100,150,200],
    #             }},
    
    # 'DecisionTreeClassifier':
    #     {
    #         'name': 'DecisionTreeClassifier',
    #         'estimator': DecisionTreeClassifier(),
    #         'params':
    #             {
    #                 'classifier__max_features': [0.1, 0.3, 0.5],
    #                 'classifier__max_depth': [5, 6, 7, 10, 15],
    #                 # 'classifier__criterion': ['gini'],
    #                 # 'classifier__max_leaf_nodes': [50, 100],
    #                 # 'classifier__min_weight_fraction_leaf': [0, 1,10,100],
    #                 # 'classifier__min_samples_split': [0.1, 1,2,10,100],
    #
    #                 'selector__k': [40],
    #             }},
    
    # 'LogisticRegression':
    #     {
    #         'name': 'LogisticRegression',
    #         'estimator': LogisticRegression(),
    #         'params':
    #             {
    #                 "classifier__class_weight": [None],
    #                 "classifier__C": [1],
    #                 "classifier__max_iter": [200],
    #                 "classifier__solver": ['lbfgs'],
    #                 "classifier__tol": [0.00001],
    #                 'selector__k': [150],
    #                 'decomposition__n_components': [110, 120, 130, 140, 150],
    #             }},
    
    # 'XGBoostClassifier':
    #     {
    #         'name': 'XGBoostClassifier',
    #         'estimator': XGBClassifier(),
    #         'params':
    #             {
    #                 'classifier__n_estimators': [100, 200],
    #                 'classifier__max_depth': [10, 50, 100],
    #                 'classifier__gamma': [1],
    #                 'classifier__reg_alpha': [0],
    #                 'classifier__reg_lambda': [0.2],
    #             }},
    #
    #     'ExtraTreesClassifier':
    #         {
    #             'name': 'ExtraTreesClassifier',
    #             'estimator': ExtraTreesClassifier(),
    #             'params':
    #                 {
    #                     'classifier__n_estimators': [100,200],
    #                     'classifier__max_depth': [10,50,100],
    #                     'classifier__gamma':[1],
    #                     'classifier__reg_alpha': [0],
    #                     'classifier__reg_lambda': [0.2],
    #                 }},
    #
    #     'AdaBoostClassifier':
    #         {
    #             'name': 'AdaBoostClassifier',
    #             'estimator': AdaBoostClassifier(),
    #             'params':
    #                 {
    #                     'classifier__n_estimators': [100,200],
    #                     'classifier__max_depth': [10,50,100],
    #                     'classifier__gamma':[1],
    #                     'classifier__reg_alpha': [0],
    #                     'classifier__reg_lambda': [0.2],
    #                 }},
    #
    'SVC':
        {
            'name': 'SVC',
            'estimator': SVC(),
            'params':
                {
                    "classifier__kernel": ["poly"],
                    "classifier__probability": [True],
                    # "classifier__degree": [1, 2, 3],
                    "classifier__C": [0.1, 1, 10],
                    # 'selector__k': [40, 50, 60],
                }},
}
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)


def get_best_classsifier(preprocess_pipeline, X_train, X_val, X_test, y_train, y_test, y_val):
    # TODO make preprocessing adasyn etc before the loop, not to repeat this fo each estimators
    # final_processing_pipe = Pipeline([
    #     ('preprocessing', preprocess_pipeline),
    #     ('sampling', ADASYN(random_state=55)),
    #     ('selector', SelectKBest(k=k_best)),
    #     ('decomposition', PCA()),
    # ])
    
    scores = {}  # dataframe
    models = {}
    
    for key, value in classifiers.items():
        tmp_pipe = Pipeline([
            # ('final_processing', final_processing_pipe),
            ('preprocessing', preprocess_pipeline),
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

        scores[value['name']] = {
            'best_params': grid.best_params_,
            'roc_auc_score_train': roc_auc_score_train,
            'roc_auc_score_val': roc_auc_score_val,
            'roc_auc_score_test': roc_auc_score_test,
        }

        models[key] = grid.best_estimator_

        #
        # # Saving models to files
        #
        from joblib import dump

        dump(grid.best_estimator_, f'models/{key}_model.joblib')

        print(f'{key} has been processed')

    return scores, models
