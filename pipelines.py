from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

# Importing inner modules
import utils
import transformers

# # # #
# # # Binary features
# #
#
# WHAT: should I prepare data in case of NaN?

binary_features = ['Gender', 'Mobile_Verified', 'Filled_Form', 'Device_Type']
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
# WHAT a lot of values to normalize is set to '-1' which gives negative results after normalization
# TODO probably should be changed to mean or mode
# TODO check if standard scaler or normalization is better for the data

numerical_features = ['Loan_Amount_Applied', 'Loan_Tenure_Applied', 'Var5',
                      'Processing_Fee', 'Interest_Rate', 'Monthly_Income']
num_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(numerical_features)),
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),
    ("back_to_df", utils.BackToDf(numerical_features))
])

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

# # # #
# # # DOB feature
# #
#
dob_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['DOB'])),
    ("dob_to_age", transformers.DobToAge()),
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),
    ("back_to_df", utils.BackToDf('Age'))
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
    ("scaler", StandardScaler(with_mean=False)),
])

# # # #
# # # City feature
# #
#
city_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['City'])),
    ("city", transformers.City()),
    # maps cities names to labels based on the frequency and fill na with most frequent label
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
    ("scaler", StandardScaler()),
    ("to_df", utils.BackToDf(['Monthly_Income'])),
])


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
        ("city_pipeline", city_pipeline),
        ("source_pipeline", source_pipeline),
        ("income_pipeline", income_pipeline),
        ("cat_pipeline", cat_pipeline),
        ("num_pipeline", num_pipeline),
    ])
    
    # # for testing
    # X_train_prep_filled2 = preprocess_pipeline.fit_transform(X_train)
    
    return preprocess_pipeline
