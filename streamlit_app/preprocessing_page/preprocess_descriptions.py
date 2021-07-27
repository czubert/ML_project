import pandas as pd
import streamlit as st

TEXT = "Below you can find the solution, and samples before and after preprocessing in DataFrames:"
def get_numeric_description():
    """
    Shows description of numeric data in streamlit
    """
    features = ['Loan_Amount_Applied', 'Loan_Tenure_Applied', 'Var5',
                'Processing_Fee', 'Interest_Rate', 'Monthly_Income']
    numerical_data_exp = st.beta_expander(label='Show numerical features preprocessing')
    with numerical_data_exp:
        st.markdown("Numeric data has a lot outliers and it's distribution is not normal."
                    "for the better functioning of the estimators,"
                    "one should try to bring the data to a normal distribution as most of them (without trees) "
                    "are prepared to work on a normal schedule."
                    )
        st.markdown("Below you can find the solution, results on a barplot, and data:")
        
        st.code(numeric_features, language='python')
        
        cols = st.beta_columns(2)
        with cols[0]:
            st.markdown('#### Before:')
            raw_data = "utils/raw_data.png"
            st.image(raw_data, caption=None, width=None, use_column_width='always', output_format='PNG')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/X.csv', usecols=features).head(50))
        
        with cols[1]:
            st.markdown('#### After:')
            normal_dist = "utils/normal_dist.png"
            st.image(normal_dist, caption=None, width=None, use_column_width='always', output_format='PNG')
            st.write(
                pd.read_csv('streamlit_app/preprocessing_page/dfs/numeric_df.csv', index_col='Unnamed: 0').head(50))

numeric_features = '''
# # # #
# # # Numerical features
# #
#
numerical_features = ['Loan_Amount_Applied', 'Loan_Tenure_Applied', 'Var5',
                      'Processing_Fee', 'Interest_Rate', 'Monthly_Income']

num_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(numerical_features)),  # Selecting numeric features to preprocess
    ("impute", SimpleImputer(strategy="median")),  # Fill NaN values with median
    ("remove_outliers", RobustScaler(quantile_range=(5.0, 95.0))),  # Getting rid of outliers
    ("box_cox", PowerTransformer(standardize=True)),  # Making data more Gaussian-like
    ("back_to_df", utils.BackToDf(numerical_features))  # making DataFrame back from NumPy array
])
'''


def get_binary_description():
    """
    Shows description of binary data in streamlit
    """
    features = ['Gender', 'Mobile_Verified', 'Filled_Form', 'Device_Type', 'Var5']
    numerical_data_exp = st.beta_expander(label='Show binary features preprocessing')
    with numerical_data_exp:
        st.markdown("Data must be numeric not string, therefore one need to change two type values to binary ones")
        st.markdown("Below you can find the solution, and samples before and after preprocessing in DataFrames:")
    
        st.code(binary_features, language='python')
    
        cols = st.beta_columns(2)
        with cols[0]:
            st.markdown('#### Before:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/X.csv', usecols=features).head(50))
    
        with cols[1]:
            st.markdown('#### After:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/binary_df.csv', index_col='Unnamed: 0').head(50))


binary_features = """
# # # #
# # # Binary features
# #
#
binary_features = ['Gender', 'Mobile_Verified', 'Filled_Form', 'Device_Type', 'Var5']
binary_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(binary_features)),  # Selecting numeric features to preprocess
    ("impute", SimpleImputer(strategy="most_frequent")),  # Fill NaN values with most frequent value
    ("back_to_df", utils.BackToDf(binary_features)),  # making DataFrame back from NumPy array
    ("bin", transformers.BinaryEncoder()),  # takes df with features of two value types and returns df with binary values
])
"""


def get_categorical_description():
    """
    Shows description of binary data in streamlit
    """
    features = ['Var1', 'Var2', 'Var4']
    categorical_data_exp = st.beta_expander(label='Show categorical features preprocessing')
    with categorical_data_exp:
        st.markdown("Data must be numeric not string, therefore one need to change nominal "
                    "and categorical type values to binary ones")
        st.markdown(TEXT)
        
        st.code(categorical_features, language='python')
        
        cols = st.beta_columns([1, 3])
        with cols[0]:
            st.markdown('#### Before:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/X.csv', usecols=features).head(50))
        
        with cols[1]:
            st.markdown('#### After:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/binary_df.csv', index_col='Unnamed: 0').head(50))


categorical_features = """
# # # #
# # # Categorical features
# # Replaces NaN with the most frequent value. Then OneHotEncoding is dividing data
#
categorical_features = ['Var1', 'Var2', 'Var4']
cat_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(categorical_features)),  # Selecting categorical features to preprocess
    ("impute", utils.MostFreqImputer()),  # Fill NaN values with most frequent value
    ("cat_encoder", utils.MyOneHotEncoder()),  # takes df with categorical features and returns df with binary values
])
"""


def get_dob_description():
    """
    Shows description of day of birth (DOB) feature in streamlit
    """
    numerical_data_exp = st.beta_expander(label='Show Day of Birth (DOB) Features Preprocessing')
    with numerical_data_exp:
        st.markdown("Data must be numeric not date_time - day of birth, therefore one need to date to for example age")
        st.markdown(TEXT)
    
        st.code(dob_features, language='python')
    
        cols = st.beta_columns([1, 5])
        with cols[0]:
            st.markdown('#### Before:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/X.csv', usecols=['DOB']).head(50))
    
        with cols[1]:
            st.markdown('#### After:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/dob_df.csv', index_col='Unnamed: 0').head(50))


dob_features = """
# # # #
# # # DOB feature
# #
#
dob_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['DOB'])),  # Selecting day of birth feature to preprocess
    ("dob_to_age", transformers.DobToAge()),  # converts DOB to Age | drops DOB
    ("impute", SimpleImputer(strategy="median")),  # Fill NaN values with median
    ("back_to_df", utils.BackToDf('Age'))  # making DataFrame back from NumPy array
])
"""


def get_submitted_description():
    """
    Shows description of 'submitted' features in streamlit
    """
    features = ['EMI_Loan_Submitted', 'Loan_Amount_Submitted', 'Loan_Tenure_Submitted', 'Processing_Fee']
    
    numerical_data_exp = st.beta_expander(label='Show "Submitted" features preprocessing')
    with numerical_data_exp:
        st.markdown("Too many values missing, therefore one cannot use imputer to fill NaN with median or mean. "
                    "The solution chosen was to set missing values to 1 and not missing to 0")
        st.markdown(TEXT)
        
        st.code(submitted_features, language='python')
        
        cols = st.beta_columns([4, 1])
        with cols[0]:
            st.markdown('#### Before:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/X.csv', usecols=features).head(50))
        
        with cols[1]:
            st.markdown('#### After:')
            st.write(
                pd.read_csv('streamlit_app/preprocessing_page/dfs/submitted_df.csv', index_col='Unnamed: 0').head(50))


submitted_features = """
# # # #
# # # Submitted feature
# #
#
submitted_features = ['EMI_Loan_Submitted', 'Loan_Amount_Submitted', 'Loan_Tenure_Submitted', 'Processing_Fee']
submitted_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(submitted_features)),  # Selecting "Submitted" features to preprocess
    ("submitted", transformers.Submitted()),  # creates Submitted_Missing feature and sets it to 1 if Submitted was missing else 0 | Original variable Loan_Amount_Submitted dropped
    ("impute", SimpleImputer(strategy="median")),  # Fill NaN values with median
])
"""


def get_city_description():
    """
    Shows description of 'submitted' features in streamlit
    """
    numerical_data_exp = st.beta_expander(label='Show City feature preprocessing')
    with numerical_data_exp:
        st.markdown("Too many different values -> grouped into 4 groups depending on the counts")
        st.markdown(TEXT)
    
        st.code(city_feature, language='python')
    
        cols = st.beta_columns([1, 5])
        with cols[0]:
            st.markdown('#### Before:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/X.csv', usecols=['City']).head(50))
    
        with cols[1]:
            st.markdown('#### After:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/city_df.csv', index_col='Unnamed: 0').head(50))


city_feature = """
# # # #
# # # City feature
# #
#
city_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['City'])),  # Selecting City feature to preprocess
    ("city", transformers.City()),  # maps cities names to labels based on the frequency and fill na with most frequent label
    ("cat_encoder", utils.MyOneHotEncoder()),  # OHE that returns dataframe with values as feature names
])
"""


def get_salary_description():
    """
    Shows description of 'submitted' features in streamlit
    """
    numerical_data_exp = st.beta_expander(label='Show Salary Feature Preprocessing')
    with numerical_data_exp:
        st.markdown("Too many different values, one need to replace less common values with 'Other', and then "
                    "OneHotEncode as values are strings")
        st.markdown(TEXT)
    
        st.code(salary_feature, language='python')
    
        cols = st.beta_columns([1, 3])
        with cols[0]:
            st.markdown('#### Before:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/X.csv', usecols=['Salary_Account']).head(50))
    
        with cols[1]:
            st.markdown('#### After:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/salary_df.csv', index_col='Unnamed: 0').head(50))


salary_feature = """
# # # #
# # # Salary_Account feature
# #
#
salary_acc_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['Salary_Account'])),  # Selecting Salary feature to preprocess
    ("salary_acc", transformers.SalaryAcc()),  # changes not common values to 'Other'
    ("impute", SimpleImputer(strategy="most_frequent")),  # Fill NaN values with most frequent value
    ("cat_encoder", utils.MyOneHotEncoder()),  # OHE that returns dataframe with feature names
])
"""


def get_employer_description():
    """
    Shows description of 'submitted' features in streamlit
    """
    numerical_data_exp = st.beta_expander(label='Show Employers Name feature preprocessing')
    with numerical_data_exp:
        st.markdown("Too many different values, one need to replace less common values with 'Other', and then "
                    "OneHotEncode as values are strings")
        st.markdown(TEXT)
    
        st.code(employer_feature, language='python')
    
        cols = st.beta_columns([2, 3])
        with cols[0]:
            st.markdown('#### Before:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/X.csv', usecols=['Employer_Name']).head(50))
    
        with cols[1]:
            st.markdown('#### After:')
            st.write(
                pd.read_csv('streamlit_app/preprocessing_page/dfs/employer_df.csv', index_col='Unnamed: 0').head(50))


employer_feature = """
# # # #
# # # Employer_Name feature
# #
#
emp_name = Pipeline([
    ("select_cat", utils.DataFrameSelector(['Employer_Name'])),  # Selecting Emplyer Name feature to preprocess
    ("salary_acc", transformers.EmpName()),  # changes not common values to 'Other'
    ("impute", SimpleImputer(strategy="most_frequent")),  # Fill NaN values with most frequent value
    ("cat_encoder", utils.MyOneHotEncoder()),  # OHE that returns dataframe with feature names
])
"""


def get_source_description():
    """
    Shows description of 'submitted' features in streamlit
    """
    numerical_data_exp = st.beta_expander(label='Show Source feature preprocessing')
    with numerical_data_exp:
        st.markdown(
            "Two values are the most popular ones (S122 and S133) the rest are in minority and changed to 'other'")
        st.markdown(TEXT)
    
        st.code(source_feature, language='python')
    
        cols = st.beta_columns([1, 6])
        with cols[0]:
            st.markdown('#### Before:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/X.csv', usecols=['Source']).head(50))
    
        with cols[1]:
            st.markdown('#### After:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/source_df.csv', index_col='Unnamed: 0').head(50))


source_feature = """
# # # #
# # # Source feature
# #
#
source_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['Source'])),  # Selecting Source feature to preprocess
    ("source", transformers.Source()),  # replaces other then 2 most popular values (S122 and S133) with 'other'
    ("impute", SimpleImputer(strategy="most_frequent")),  # Fill NaN values with most frequent value
    ("to_df", utils.BackToDf(['Source'])),  # making DataFrame back from NumPy array
    ("cat_encoder", utils.MyOneHotEncoder()),  # takes df with categorical features and returns df with binary values
])
"""


def get_income_description():
    """
    Shows description of 'submitted' features in streamlit
    """
    numerical_data_exp = st.beta_expander(label='Show Income Feature Preprocessing')
    with numerical_data_exp:
        st.markdown("Income is numeric, however there are some outliers, therefore one should change them to "
                    "95th percentile (according to the plot)")
        st.markdown(TEXT)
        
        st.code(income_feature, language='python')
        
        cols = st.beta_columns([1, 6])
        with cols[0]:
            st.markdown('#### Before:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/X.csv', usecols=['Monthly_Income']).head(50))
        
        with cols[1]:
            st.markdown('#### After:')
            st.write(pd.read_csv('streamlit_app/preprocessing_page/dfs/income_df.csv', index_col='Unnamed: 0').head(50))


income_feature = """
# # # #
# # # Income features
# #
#
income_pipeline = Pipeline([
    ("select_cat", utils.DataFrameSelector(['Monthly_Income'])), # Selecting Monthly Income feature to preprocess
    ("income", transformers.Income()),  # changes outliers to the value of 95th percentile
    ("impute", SimpleImputer(strategy="median")),  # Fill NaN values with median
    ("to_df", utils.BackToDf(['Monthly_Income'])),  # making DataFrame back from NumPy array
])
"""
