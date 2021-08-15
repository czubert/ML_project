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

# # # #
# # # Categorical features
# # Replaces NaN with the most frequent value. Then OneHotEncoding is dividing data
#
categorical_features = ['Var1', 'Var2', 'Var4']
cat_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(categorical_features)),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("cat_encoder", utils.MyOneHotEncoder()),
])

# # # #
# # # DOB feature
# # converts DOB to Age | drops DOB
#
dob_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['DOB', 'Lead_Creation_Date'])),
    ("dob_to_age", transformers.DobToAge()),
    ("impute", SimpleImputer(strategy="median")),
    ("back_to_df", utils.BackToDf(['Age']))
])

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


def get_preprocessed_data():
    """
    Concatenates all the preprocessed data into one array and returns it
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

    return preprocess_pipeline
