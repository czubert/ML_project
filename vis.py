import streamlit as st

from streamlit_app import sidebar, utils, pages


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
    
    tabs = st.sidebar.radio("", ('Main', 'Data Profiling', 'Data Preprocessing', 'Predictions'))

    if tabs == 'Main':
        # # #
        # # Main Page
        #
        pages.show_main_content(df)

    if tabs == "Data Profiling":
        # # #
        # # Profiling page
        #
        pages.show_data_profile()

    if tabs == "Data Preprocessing":
        # # #
        # # Profiling page
        #
        pages.show_data_preprocessing()

    if tabs == "Predictions":
        # # #
        # # Machine Learning part
        #
        pages.show_predictions_page(df)
    


if __name__ == '__main__':
    try:
        main()
    except KeyError:
        "Something went wrong"
    
    print("Streamlit finished it's work")
