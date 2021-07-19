import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from joblib import load

from streamlit_app import utils, descriptions, fake_data


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
    data_exp = st.beta_expander(label='Original data')
    with data_exp:
        st.write(df)
    
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


def run_predictions_page(df):
    # containter to prepare fake data
    fake_data_exp = st.beta_expander('Create your own data for predictions')
    with fake_data_exp:
        fake_df = pd.DataFrame(fake_data.create_fake_data(df), index=['Fake data'])
        st.write(fake_df)
        
        model = load('models/LogisticRegression_model.joblib')
        st.write(model.predict(fake_df))
    
    estimators = ['RandomForestClassifier', 'DecisionTreeClassifier', 'LogisticRegression',
                  'XGBoostClassifier', 'SVC']
    
    options = st.multiselect(
        'What are your favorite colors',
        estimators)
    st.write('You selected:', options)

# from joblib import load
# model = load('models/LogisticRegression_model.joblib')
# model.predict(X_test)
