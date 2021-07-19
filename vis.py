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
    
    tabs = st.sidebar.radio("", ('Main', 'Data Profiling', 'Predictions'))

    if tabs == 'Main':
        # # #
        # # Main Page
        #
        pages.show_main_content(df)

    elif tabs == "Data Profiling":
        # # #
        # # Profiling page
        #
        pages.show_data_profile()

    elif tabs == "Predictions":
        # # #
        # # Machine Learning part
        #
        st.sidebar.file_uploader('Upload data for tests')
        pages.run_predictions_page(df)
    
    
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
