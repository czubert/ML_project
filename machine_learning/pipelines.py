import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PowerTransformer, RobustScaler

from machine_learning import utils, transformers

# # # #
# # # Binary features
# #
#
binary_features = ['Gender', 'Mobile_Verified', 'Filled_Form', 'Device_Type', 'Var5']
binary_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(binary_features)),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("back_to_df", utils.BackToDf(binary_features)),
    ("bin", transformers.BinaryEncoder()),  # takes df with features of two values and returns df with binary values
])
# pd.DataFrame(binary_pipeline.fit_transform(X_train)).to_csv('binary_df.csv')
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
# pd.DataFrame(num_pipeline.fit_transform(X_train)).to_csv('numeric_df.csv')
# # # #
# # # Categorical features
# # Replaces NaN with the most frequent value. Then OneHotEncoding is dividing data
#
categorical_features = ['Var1', 'Var2', 'Var4']
cat_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(categorical_features)),
    ("impute", utils.MostFreqImputer()),
    ("cat_encoder", utils.MyOneHotEncoder()),
])
# pd.DataFrame(cat_pipeline.fit_transform(X_train)).to_csv('categorical_df.csv')

# # # #
# # # DOB feature
# # converts DOB to Age | drops DOB
#
dob_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['DOB'])),
    ("dob_to_age", transformers.DobToAge()),
    ("impute", SimpleImputer(strategy="median")),
    ("back_to_df", utils.BackToDf(['Age']))
])
# pd.DataFrame(dob_pipeline.fit_transform(X_train)).to_csv('dob_df.csv')

# # # #
# # # Submitted feature
# #
#
submitted_features = ['EMI_Loan_Submitted', 'Loan_Amount_Submitted', 'Loan_Tenure_Submitted', 'Processing_Fee']
submitted_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(submitted_features)),
    ("submitted", transformers.Submitted()),
    ("impute", SimpleImputer(strategy="median")),
])
# pd.DataFrame(submitted_pipeline.fit_transform(X_train)).to_csv('submitted_df.csv')

# # # #
# # # City feature
# #
#
city_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['City'])),
    # maps cities names to labels based on the frequency and fill na with most frequent label
    ("city", transformers.City()),
    ("cat_encoder", utils.MyOneHotEncoder()),  # OHE that returns dataframe with feature names
])
# pd.DataFrame(city_pipeline.fit_transform(X_train)).to_csv('city_df.csv')

# # # #
# # # Salary_Account feature
# #
#
salary_acc_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['Salary_Account'])),
    ("salary_acc", transformers.SalaryAcc()),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("cat_encoder", utils.MyOneHotEncoder()),  # OHE that returns dataframe with feature names
])
# pd.DataFrame(salary_acc_pipeline.fit_transform(X_train)).to_csv('salary_df.csv')

# # # #
# # # Employer_Name feature
# #
#
emp_name = Pipeline([
    ("select_cat", utils.DataFrameSelector(['Employer_Name'])),
    ("salary_acc", transformers.EmpName()),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("cat_encoder", utils.MyOneHotEncoder()),  # OHE that returns dataframe with feature names
])
# pd.DataFrame(emp_name.fit_transform(X_train)).to_csv('employer_df.csv')

# # # #
# # # Source feature
# #
#
source_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['Source'])),
    ("source", transformers.Source()),  # replaces other then 2 most popular values with 'other'
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("to_df", utils.BackToDf(['Source'])),
    ("cat_encoder", utils.MyOneHotEncoder()),
])
# pd.DataFrame(source_pipeline.fit_transform(X_train)).to_csv('source_df.csv')

# # # #
# # # Income features
# #
#
income_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['Monthly_Income'])),
    ("income", transformers.Income()),
    ("impute", SimpleImputer(strategy="median")),
    ("to_df", utils.BackToDf(['Monthly_Income'])),
])


# pd.DataFrame(income_pipeline.fit_transform(X_train)).to_csv('income_df.csv')


def get_preprocessed_data(X_train=None):
    """
    Concatenates all the preprocessed data into one array and returns it
    :param X_train: Training dataset
    :return: Numpy array with all preprocessed data
    """
    preprocess_pipeline = FeatureUnion(transformer_list=[
        ("bin_pipeline", binary_pipeline),
        ("dob_pipeline", city_pipeline),
        ("submitted_pipeline", submitted_pipeline),
        ("salary_pipeline", salary_acc_pipeline),
        ("emp_name", emp_name),
        ("city_pipeline", city_pipeline),
        ("source_pipeline", source_pipeline),
        ("income_pipeline", income_pipeline),
        ("cat_pipeline", cat_pipeline),
        ("num_pipeline", num_pipeline),
    ])

    # # for testing
    # X_train_prep_filled2 = preprocess_pipeline.fit_transform(X_train)
    # colnames = preprocess_pipeline.get_feature_names()

    return preprocess_pipeline
