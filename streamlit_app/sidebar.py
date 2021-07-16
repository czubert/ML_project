import base64
import streamlit as st


@st.cache
def show_logo():
    with open('streamlit_app/media/newlogo_nobg.png', 'rb') as f:
        data = f.read()
    
    bin_str = base64.b64encode(data).decode()
    html_code = f'''
                    <img src="data:image/png;base64,{bin_str}"
                    style="
                         margin: auto;
                         margin-top:-40px;
                         margin-left:15%;
                         width: 50%;
                         padding:0px 6px 0px 20%;
                         "/>
                '''
    return html_code
