# # #
# # # # Machine Learning Part
# # #

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

# classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# imblearn
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN

# # CV
seed = 123
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

classifiers = {
    'RF':
        {
            'name': 'Random Forest Classifier',
            'estimator': RandomForestClassifier(),
            'params':
                {
                    # 'classifier__n_estimators': [10,100,400],
                    # 'classifier__criterion' :['gini', 'entropy'],
                    # 'classifier__max_features': [0.25,0.5,0.75],
                    'classifier__max_depth': [8],
                    # 'selector__k' :[40,50,54],
                }},
    
    #     'tree':
    #         {
    #             'name': 'Decision Tree Classifier',
    #             'estimator': DecisionTreeClassifier(),
    #             'params':
    #                 {
    #                     'classifier__max_features': [0.25, 0.5]
    #                 }},
}


def get_best_classsifier(preprocess_pipeline, X_train, y_train, X_test, y_test, k_best):
    # final_processing_pipe = Pipeline([
    #     ('preprocessing', preprocess_pipeline),
    #     ('sampling', ADASYN(random_state=55)),
    #     ('selector', SelectKBest(k=k_best)),
    #     ('decomposition', PCA()),
    # ])
    
    final_report = {}
    models = {}
    
    for classifier in classifiers:
        tmp_pipe = Pipeline([
            # ('final_processing', final_processing_pipe),
            ('preprocessing', preprocess_pipeline),
            ('sampling', ADASYN(random_state=55)),
            ('selector', SelectKBest(k=k_best)),
            ('decomposition', PCA()),
            ('classifier', classifiers[classifier]['estimator'])
        ])
        
        grid = GridSearchCV(tmp_pipe, classifiers[classifier]['params'], cv=kfold)
        grid.fit(X_train, y_train)
        
        # Show the classification report
        print(f'')
        print(f'Classfier:\n{classifiers[classifier]["name"]}\n'
              f'Best params:\n'
              f'{grid.best_params_}\n'
              f'{classifiers[classifier]["name"]} performance report:\n'
              f'{classification_report_imbalanced(y_test, grid.best_estimator_.predict(X_test))}')
        
        report = classification_report_imbalanced(y_test, grid.best_estimator_.predict(X_test))
        
        final_report[classifiers[classifier]['name']] = {'best_params': grid.best_params_, 'report': report}
        models[classifiers[classifier]['name']] = [grid]
    
    return final_report, models
