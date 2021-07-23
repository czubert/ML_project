import streamlit as st
import datetime


def create_customer_data(df):
    fake_data_container = st.beta_container()
    with fake_data_container:
        cols = st.beta_columns(7)
        with cols[0]:
            fake_id = st.text_input('Input your ID', value='ID001050K00')  # ID
            fake_lar = st.number_input('Loan Amount Requested (INR)')  # Loan_Amount_Applied
            fake_var5 = st.selectbox('Select Var5', df.Var5.unique())  # Var5
            fake_els = st.number_input('Input EMI Loan Submitted', step=500)  # EMI_Loan_Submitted
            fake_logged = st.selectbox('LoggedIn?', df.LoggedIn.unique())  # LoggedIn
        
        with cols[1]:
            fake_gender = st.selectbox('Select Your Gender', df.Gender.unique())  # Gender
            fake_ltr = st.selectbox('Select Loan Tenure Requested (in years)',
                                    df.Loan_Tenure_Applied.unique())  # Loan_Tenure_Applied
            fake_var1 = st.selectbox('Select Var1', df.Var1.unique())  # Var1
            fake_form = st.selectbox('Filled Form?', df.Filled_Form.unique())  # Filled_Form
        
        with cols[2]:
            fake_city = st.selectbox('Select your city', df.City.unique())  # City
            fake_emi = st.number_input('Provide EMI of Existing Loans (INR)')  # Existing_EMI
            fake_las = st.number_input('Provide Loan Amount Submitted')  # Loan_Amount_Submitted
            fake_device = st.selectbox('Select Device Type', df.Device_Type.unique())  # Device_Type
        
        with cols[3]:
            fake_income = st.number_input('Input Your Monthly Income', step=1000)  # Monthly_Income
            fake_employer = st.selectbox('Select Your Emplyers Name', df.Employer_Name.unique())  # Employer_Name
            fake_lts = st.selectbox('Select Loan Tenure Submitted',
                                    df.Loan_Tenure_Submitted.dropna().unique())  # Loan_Tenure_Submitted
            fake_var2 = st.selectbox('Select Var2', df.Var2.unique())  # Var2
        
        with cols[4]:
            fake_dob = st.date_input('Input Date of Birth', datetime.date(1985, 7, 6))  # DOB
            fake_salary = st.selectbox('Input Your Salary Account', df.Salary_Account.unique())  # Salary_Account
            # TODO important - check if Interest_Rate is for sure numerical not categorical
            fake_interest = st.slider('Select Your Salary Account', 11., 40., 20., 0.5)  # Interest_Rate
            fake_source = st.selectbox('Select Source', df.Source.unique())  # Source
        
        with cols[5]:
            fake_lcd = st.date_input('Input Lead Created date', datetime.date(2015, 7, 6))  # Lead_Creation_Date
            fake_mobile = st.selectbox('Mobile verified?', df.Mobile_Verified.unique())  # Mobile_Verified
            fake_fee = st.number_input('Input Processing Fee', step=500)  # Processing_Fee
            fake_var4 = st.selectbox('Select Var4', df.Var4.unique())  # Var4
        
        fake_examples = [fake_id, fake_gender, fake_city, fake_income, fake_dob, fake_lcd, fake_lar, fake_ltr, fake_emi,
                         fake_employer,
                         fake_salary, fake_mobile, fake_var5, fake_var1, fake_las, fake_lts, fake_interest, fake_fee,
                         fake_els, fake_form,
                         fake_device, fake_var2, fake_source, fake_var4, fake_logged]
        
        fake_data = dict(zip(df, fake_examples))
    
    return fake_data
