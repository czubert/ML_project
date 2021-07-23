import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from joblib import load

from streamlit_app import utils, descriptions, fake_data_preparation
from streamlit_app.preprocessing_page import preprocess_descriptions


def show_main_content(df):
    """
    Showing main page in streamlit
    :param df: DataFrame with raw data
    """
    
    #
    # # Project logo
    #
    html_code = utils.show_logo()
    st.markdown(html_code, unsafe_allow_html=True)
    
    #
    # # Problem Description
    #
    outcome_exp = st.beta_expander(label='Task description')
    with outcome_exp:
        for i, el in enumerate(descriptions.description_list):
            st.markdown(f'<p class="descr">{el}</p>', unsafe_allow_html=True)
        for i, el in enumerate(descriptions.data_dscr_list):
            st.markdown(f'<p class="font">{el}</p>', unsafe_allow_html=True)
    
    #
    # # Preprocessing Description
    #
    data_exp = st.beta_expander(label='Original data representation')
    with data_exp:
        st.write(df.head(100))
    
    #
    # # Preprocessing Description
    #
    preprocess_exp = st.beta_expander(label='Data preprocessing description')
    with preprocess_exp:
        for el in descriptions.preprocessing_descr_list:
            st.markdown(f'<p class="font">{el}</p>', unsafe_allow_html=True)
    
    #
    # # results obtained by LazyPredict
    #
    lazypredict_exp = st.beta_expander(label='Lazy Predict scores')
    with lazypredict_exp:
        df_fast_results = pd.read_csv('models.csv')
        cols = df_fast_results.columns[:-1]
        df_fast_results['std'] = round(df_fast_results[cols].std(axis=1), 7)
        st.write(df_fast_results)


def show_data_profile():
    """
    Showing data profiling. Tools used: pandas-profiling report, Plotly scatter matrix and seaborn pairplot
    """
    st.header('Data profiling')
    
    profiling_exp = st.beta_expander(label='Show pandas-profiling report')
    with profiling_exp:
        HtmlFile = open("downloads/pandas_report.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=5000, scrolling=True)
    
    # TODO make it more clear
    scatter_matrix_exp = st.beta_expander(label='Show Plotly scatter matrix')
    with scatter_matrix_exp:
        HtmlFile = open("downloads/pxplot.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=5000, scrolling=True)
    
    pairplot_exp = st.beta_expander(label='Show pairplot')
    with pairplot_exp:
        image = "downloads/snspairplot.png"
        st.image(image, caption=None, width=None, use_column_width='always', output_format='auto')


def show_data_preprocessing():
    """
    Showing page with preprocessing description and data before and after preprocessing
    """
    st.header('Data Preprocessing Description ')
    preprocess_descriptions.get_numeric_description()
    preprocess_descriptions.get_binary_description()
    preprocess_descriptions.get_categorical_description()
    preprocess_descriptions.get_dob_description()
    preprocess_descriptions.get_submitted_description()
    preprocess_descriptions.get_city_description()
    preprocess_descriptions.get_salary_description()
    preprocess_descriptions.get_employer_description()
    preprocess_descriptions.get_source_description()
    preprocess_descriptions.get_income_description()


def show_predictions_page(df):
    """
    Showing predictions page. This page consists of a section to create customer profile (data), to predict whether
    he should get the 'Disburse' or not.
    Also, you can choose between customer data, example data provided by the bank, and you can upload your own data.
    You can choos which estimator/estimators would you like to use to predict the 'Disburse'.
    In last section you can see the scores for train/validation/test data for chosen estimators
    and also it's learning parameters.
    :param df: Raw data provided by the bank
    """
    
    # # # #
    # # #  part responsible for getting the data
    # #
    #
    
    #
    # # Choosing the source of the data
    #
    possible_data = ['Customer data', 'Example data', 'Uploaded data']
    st.markdown('#### Choose what data would you like to use for predictions')
    chosen_data = st.radio('', possible_data)
    
    if chosen_data == 'Customer data':
        # # Customer Data
        customer_data_exp = st.beta_expander('Create Customer data and predict if "Disbursed"')
        with customer_data_exp:
            customer_data = pd.DataFrame(fake_data_preparation.create_customer_data(df),
                                         index=['Fake data'])  # creating customer profile data
            st.write(customer_data)  # showing Customer data
        
        processing_data = customer_data
    
    if chosen_data == 'Example data':
        # # Example Data provided by the Bank
        bank_data = pd.read_csv('data/test.csv')
        
        processing_data = bank_data
    
    if chosen_data == 'Uploaded data':
        # # Uploaded Data provided by the user
        uploaded_data = st.file_uploader('Upload data for tests')
        processing_data = uploaded_data
    
    # # # #
    # # #  Part responsible for managing the data, includes results and scores
    # #
    #
    
    # # Importing Scores
    scores = pd.read_csv('models/scores.csv', index_col='Unnamed: 0')
    
    # # All available models
    estimators = scores.columns
    st.markdown("---")
    
    # # Choosing one or more models for predictions (only estimators that have calculated scores)
    st.markdown('#### Select trained models to use for predictions')
    chosen_estimators = st.multiselect('', estimators)
    
    # # Estimating the "Disbursed" feature
    if chosen_estimators:  # Works only if any estimator is selected
        st.markdown("##### Predictions for you: ")
        st.markdown('')
        
        best_params_of_chosen_estimators = []
        
        for chosen_estimator in chosen_estimators:
            model = load(f'models/{"_".join(chosen_estimator.split())}_model.joblib')
            st.write(f'{chosen_estimator} prediction: {model.predict(processing_data)[0]}')
            
            list_of_best_params = scores.loc['best_params', chosen_estimator].strip('{').strip('}').split(',')
            best_params_dict = dict()
            
            # # Getting names and values of best parameters from grid search results
            for el in list_of_best_params:
                key, value = el.split(sep=':', maxsplit=1)
                key = key.strip().strip("'")
                value = value.strip()
                best_params_dict[key] = value
            
            # # DataFrame of best params for chosen estimator
            best_params_df = pd.DataFrame(best_params_dict.values(),
                                          index=best_params_dict.keys(),
                                          columns=[chosen_estimator])
            
            best_params_of_chosen_estimators.append(best_params_df)
        
        # # DataFrame of best params for chosen estimators
        if len(best_params_of_chosen_estimators) == 1:
            final_best_params_df = best_params_of_chosen_estimators[0]
        elif len(best_params_of_chosen_estimators) > 1:
            final_best_params_df = pd.concat(best_params_of_chosen_estimators, axis=0)
        
        # # DataFrame of scores train/val/test
        scores_for_chosen_estimators = scores.loc[:, chosen_estimators].drop(['best_params'], axis=0)
        
        st.markdown("---")
        
        if any(estimators) is any(chosen_estimators):
            st.markdown("#### Chosen models were trained, validated, and tested withwith following scores:")
            st.markdown("")
            # Showing a DataFrame of scores and best params for chosen estimators
            st.write(pd.concat([scores_for_chosen_estimators, final_best_params_df], axis=0))
