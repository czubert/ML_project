# # #
# # # # Machine Learning Part
# # #

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_selection import SelectKBest
from imblearn.metrics import classification_report_imbalanced

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
    
    'tree':
        {
            'name': 'Decision Tree Classifier',
            'estimator': DecisionTreeClassifier(),
            'params':
                {
                    'classifier__max_features': [0.25, 0.5]
                }},
}


def get_best_classsifier(preprocess_pipeline, X_train, y_train, X_test, y_test):
    for classifier in classifiers:
        pipe = Pipeline([
            ('preprocessing', preprocess_pipeline),
            ('sampling', ADASYN(random_state=55)),
            ('selector', SelectKBest(k=50)),
            ('classifier', classifiers[classifier]['estimator'])
        ])
        
        grid = GridSearchCV(pipe, classifiers[classifier]['params'], cv=kfold)
        grid.fit(X_train, y_train)
        print(f'Best params:\n{grid.best_params_}')
        
        # Show the classification report
        print(
            f'Classfier: {classifier}\n{classification_report_imbalanced(y_test, grid.best_estimator_.predict(X_test))}')
