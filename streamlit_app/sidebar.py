import streamlit as st

from streamlit_app import utils


def sidebar():
    # Project logo
    html_code = utils.show_icon()
    st.sidebar.markdown(html_code, unsafe_allow_html=True)
