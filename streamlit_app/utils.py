import base64
import streamlit as st
import pandas as pd


def set_app_config():
    st.set_page_config(
        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        page_title='MachineLearning project kodolamacz bootcamp',  # String or None.
        page_icon="streamlit_app/media/icon.png",  # String, anything supported by st.image, or None.
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


@st.cache
def show_logo():
    with open('streamlit_app/media/3_nobg.png', 'rb') as f:
        data = f.read()
    
    bin_str = base64.b64encode(data).decode()
    html_code = f'''
                    <img src="data:image/png;base64,{bin_str}"
                    style="
                         margin: auto;
                         margin-top:-60px;
                         margin-left:15%;
                         width: 60%;
                         padding:0px 6px 0px 10%;
                         "/>
                '''
    return html_code


@st.cache
def show_icon():
    with open('streamlit_app/media/icon.png', 'rb') as f:
        data = f.read()
    
    bin_str = base64.b64encode(data).decode()
    html_code = f'''
                    <img src="data:image/png;base64,{bin_str}"
                    style="
                         margin: auto;
                         margin-top:-60px;
                         margin-left:15%;
                         width: 60%;
                         padding:0px 6px 0px 10%;
                         "/>
                '''
    return html_code


@st.cache
def get_data():
    return pd.read_csv('data/Train_nyOWmfK.csv', encoding="latin1")

    # pairplot_exp = st.beta_expander(label='Show pairplot')
    # with pairplot_exp:
    #     HtmlFile = open("downloads/snspairplot.html", 'r', encoding='utf-8')
    #     source_code = HtmlFile.read()
    #     components.html(source_code, height=5000, scrolling=True)
