from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import classes

# # # #
# # # Binary features
# #
#
binary_features = ['Gender', 'Mobile_Verified', 'Filled_Form', 'Device_Type']

binary_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(binary_features)),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("back_to_df", classes.BackToDf(binary_features)),
    ("bin", classes.BinaryEncoder()),
])
# lol = binary_pipeline.fit_transform(X_train)

# # # #
# # # Numerical features
# #
#
# WHAT a lot of values to normalize is set to '-1' which gives negative results after normalization
# TODO probably should be changed to mean or mode
# TODO check if standard scaler or normalization is better for the data

features_to_normalize = ['Loan_Amount_Submitted', 'Loan_Tenure_Submitted', 'Loan_Amount_Applied',
                         'Loan_Tenure_Applied', 'Var5', 'EMI_Loan_Submitted', 'Processing_Fee',
                         'Interest_Rate', 'Monthly_Income']
num_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(features_to_normalize)),
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),
    ("back_to_df", classes.BackToDf(features_to_normalize))
])
# # for testing
# num_pipe = num_pipeline.fit_transform(X_train)

# # # #
# # # Categorical features
# # Replaces NaN with the most frequent value. Then OneHotEncoding is dividing data
#
to_one_hot = ['Var1', 'Var2', 'Var4']
cat_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(to_one_hot)),
    ("impute", classes.MostFreqImputer()),
    ("cat_encoder", classes.MyOneHotEncoder()),
])
# # for testing
# cat_lol = cat_pipeline.fit_transform(X_train)

# # # #
# # # City feature
# #
#
city_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(['City'])),
    ("city", classes.City()),
    ("cat_encoder", classes.MyOneHotEncoder()),
])
# # for testing
# city_lol = city_pipeline.fit_transform(X_train)

# # # #
# # # Source feature
# #
#
source_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(['Source'])),
    ("source", classes.Source()),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("to_df", classes.BackToDf(['Source'])),
    ("cat_encoder", classes.MyOneHotEncoder()),
])
# # for testing
# source_lol = source_pipeline.fit_transform(X_train)

# # # #
# # # Income features
# #
#
income_pipeline = Pipeline([
    ("select_cat", classes.DataFrameSelector(['Monthly_Income'])),
    ("income", classes.Income()),
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("to_df", classes.BackToDf(['Monthly_Income'])),
])


# # for testing
# income_lol = income_pipeline.fit_transform(X_train)


def get_preprocessed_data(X_train):
    """
    Concatenates all the preprocessed data into one array and returns it
    :param X_train: Training dataset
    :return: Numpy array with all preprocessed data
    """
    preprocess_pipeline = FeatureUnion(transformer_list=[
        ("bin_pipeline", binary_pipeline),
        ("city_pipeline", city_pipeline),
        ("source_pipeline", source_pipeline),
        ("income_pipeline", income_pipeline),
        ("cat_pipeline", cat_pipeline),
        ("num_pipeline", num_pipeline),
    ])
    
    # # for testing
    # X_train_prep_filled2 = preprocess_pipeline.fit_transform(X_train)
    
    return preprocess_pipeline
