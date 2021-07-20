numerical_data = '''
# # # #
# # # Numerical features
# #
#
numerical_features = ['Loan_Amount_Applied', 'Loan_Tenure_Applied', 'Var5',
                      'Processing_Fee', 'Interest_Rate', 'Monthly_Income']

num_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(numerical_features)),
    ("impute", SimpleImputer(strategy="median")),  # TODO compare to median
    ("remove_outliers", RobustScaler(quantile_range=(5.0, 95.0))),
    ("box_cox", PowerTransformer(standardize=True)),
    ("back_to_df", utils.BackToDf(numerical_features))
])
'''
