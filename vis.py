import streamlit as st
from streamlit_app import descriptions, sidebar


def main():
    """
    Streamlit app solving the problem of categorization from Hackathon.
    :return:
    """
    
    # # # #
    # # # Streamlit settings
    # #
    #
    st.set_page_config(
        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        page_title='MachineLearning project kodolamacz bootcamp',  # String or None.
        page_icon="streamlit_app/media/icon_nobg.png",  # String, anything supported by st.image, or None.
    )
    
    st.markdown("""
                <style>
                .font {
                    font-size:13px !important;
                    line-height:8px !important;
                }
                .descr {
                    font-size:13px !important;
                    line-height:20px !important;
                }
                </style>
                """, unsafe_allow_html=True)
    
    # # # #
    # # # Main Page
    # #
    #
    
    # Project logo
    html_code = sidebar.show_logo()
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
    
    #
    # # Sidebar
    #
    st.sidebar.write('qwe')

if __name__ == '__main__':
    try:
        main()
    except KeyError:
        "Something went wrong"
    
    print("Streamlit finished it's work")
