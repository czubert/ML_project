import base64
import streamlit as st
import pandas as pd
import pandas_profiling

from streamlit_pandas_profiling import st_profile_report


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


import re
import requests
import streamlit as st

from contextlib import contextmanager
from pathlib import Path


# _filter_share = re.compile(r"^.*\[share_\w+\].*$", re.MULTILINE)


# @contextmanager
# def readme(project, usage=None, source=None):
#     # content = requests.get(f"https://raw.githubusercontent.com/okld/{project}/main/README.md").text
#     # st.markdown(_filter_share.sub("", content))
#     #
#     # demo = st.beta_container()
#
#     # if source:
#     #     with st.beta_expander("SOURCE"):
#     #         st.code(Path(source).read_text())
#     #
#     # with demo:
#     #     yield
#     st.write('lol')

# def show_data_profile():
#     # with readme("streamlit-pandas-profiling", st_profile_report, __file__):
#     dataset = 'https://www.kaggle.com/xinxinnxin/hackathon/version/1'
#
#     df = pd.read_csv('data/Train_nyOWmfK.csv', encoding="latin1")
#
#     st.write(f"🔗 [Happy Customer Bank dataset]({dataset})")
#
#     if st.button("Generate report"):
#         pr = df.profile_report()
#
#         with st.beta_expander("REPORT", expanded=True):
#             st_profile_report(pr)

def show_data_profile():
    df = pd.read_csv('data/Train_nyOWmfK.csv', encoding="latin1")
    
    if st.button("Generate report"):
        import streamlit.components.v1 as components
        
        HtmlFile = open("downloads/pandas_report.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code)
