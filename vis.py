import streamlit as st
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from streamlit_app import descriptions, sidebar, utils


def main():
    """
    Streamlit app solving the problem of categorization from Hackathon.
    """
    
    # # #
    # # Streamlit settings and styling
    #
    utils.set_app_config()
    
    # # #
    # # Get data
    #
    df = utils.get_data()
    
    # # #
    # # Show sidebar
    #
    sidebar.sidebar()
    
    tabs = st.sidebar.radio("", ('Main', 'Data Profiling', 'Predictions'))
    
    if tabs == 'Main':
        # # # Main Page
        # #
        #
        
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
        preprocess_exp = st.beta_expander(label='Show description of data preprocessing')
        with preprocess_exp:
            for el in descriptions.preprocessing_descr_list:
                st.markdown(f'<p class="font">{el}</p>', unsafe_allow_html=True)
    
    
    elif tabs == "Data Profiling":
        # # # Profiling page
        # #
        #
        utils.show_data_profile()
    
    elif tabs == "Predictions":
        # # # Machine Learning part
        # #
        #
        st.write("You didn't select comedy.")
    
    
    else:
        # # # If needed in the future
        # #
        #
        st.write('Something went wrong')


if __name__ == '__main__':
    try:
        main()
    except KeyError:
        "Something went wrong"
    
    print("Streamlit finished it's work")
