import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from joblib import load

from streamlit_app import utils, descriptions, fake_data_preparation
from streamlit_app.preprocessing_page import descriptions


def show_main_content(df):
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
    st.header('Data Preprocessing Description ')
    descriptions.get_numeric_description()
    descriptions.get_binary_description()
    descriptions.get_categorical_description()
    descriptions.get_dob_description()
    descriptions.get_submitted_description()
    descriptions.get_city_description()
    descriptions.get_salary_description()
    descriptions.get_employer_description()
    descriptions.get_source_description()
    descriptions.get_income_description()


def show_predictions_page(df):
    # Expander to prepare fake data
    fake_data_exp = st.beta_expander('Create your own data and predict if "Disbursed"')
    
    with fake_data_exp:
        fake_df = pd.DataFrame(fake_data_preparation.create_fake_data(df), index=['Fake data'])  # creating fake data
        st.write(fake_df)  # showing fake data
    
    #
    # # Data
    #
    possible_data = {'Created data': fake_df, 'Example data': fake_df, 'Uploaded data': fake_df}
    st.markdown('#### Choose what data would you like to use for predictions')
    chosen_data = st.radio('Choose what data would you like to use for predictions', possible_data)
    
    #
    # # Models
    #
    
    # # Importing Scores
    scores = pd.read_csv('scores.csv')
    
    # # All available models
    estimators = ['RandomForestClassifier', 'DecisionTreeClassifier', 'LogisticRegression',
                  'XGBoostClassifier', 'SVC', 'AdaBoostClassifier']
    st.markdown("---")
    
    # Choosing one or more models for predictions
    st.markdown('#### Select trained models to use for predictions')
    chosen_estimators = st.multiselect('', estimators)
    
    # Estimating the "Disbursed"
    if chosen_estimators is not None:
        st.markdown("##### Predictions for you: ")
        st.markdown('')
        for chosen_estimator in chosen_estimators:
            model = load(f'models/{"_".join(chosen_estimator.split())}_model.joblib')
            st.write(f'{chosen_estimator} prediction: {model.predict(possible_data[chosen_data])[0]}')
        
        st.markdown("---")
        st.markdown("#### Scores for chosen estimators on training/validation/test data:")
        st.markdown("")
        st.write(scores.loc[:, chosen_estimators])
