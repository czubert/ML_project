import streamlit as st
import descriptions


def main():
    """
    Streamlit app solving the problem of categorization from Hackathon.
    :return:
    """
    
    #
    # # Streamlit settings
    #
    st.set_page_config(
        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        page_title='MachineLearning project kodolamacz bootcamp',  # String or None.
        page_icon='ML',  # String, anything supported by st.image, or None.
    )
    
    st.markdown("""
                <style>
                .big-font {
                    font-size:12px !important;
                }
                </style>
                """, unsafe_allow_html=True)
    
    #
    # # Description of the data
    #
    outcome_exp = st.beta_expander(label='Task description')
    with outcome_exp:
        st.markdown(descriptions.link, unsafe_allow_html=True)
        
        for el in descriptions.variables_list:
            st.markdown(f'<p class="big-font">{el}</p>', unsafe_allow_html=True)
    
    preprocess_exp = st.beta_expander(label='Show description of data preprocessing')
    with preprocess_exp:
        for el in descriptions.preprocessing_descr_list:
            st.markdown(f'<p class="big-font">{el}</p>', unsafe_allow_html=True)


if __name__ == '__main__':
    try:
        
        main()
    except KeyError:
        main()
    
    print("Streamlit finished it's work")
