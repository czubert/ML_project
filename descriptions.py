about_company = 'Happy Customer Bank is a mid-sized private bank which deals in all kinds of loans. ' \
                'They have presence across all major cities in India and focus on lending products.' \
                'They have a digital arm which sources customers from the internet.'

problem = 'Digital arms of banks today face challenges with lead conversion, ' \
          'they source leads through mediums like search, display, email campaigns and via affiliate partners. ' \
          'Here Happy Customer Bank faces same challenge of low conversion ratio. ' \
          'They have given a problem to identify the customers segments having higher conversion ratio ' \
          'for a specific loan product so that they can specifically target these customers, ' \
          'here they have provided a partial data set for salaried customers only from the last 3 months. ' \
          'They also capture basic details about customers like gender, DOB, existing EMI, employer Name, ' \
          'Loan Amount Required, Monthly Income, City, Interaction data and many others. ' \
          'Let’s look at the process at Happy Customer Bank. ' \
          'In above process, customer applications can drop majorly at two stages, ' \
          'at login and approval/ rejection by bank. ' \
          'Here we need to identify the segment of customers having higher disbursal rate in next 30 days.'

link = """https://discuss.analyticsvidhya.com/t/hackathon-3-x-predict-customer-worth-for-happy-customer-bank/3802"""

data_set = 'We have train and test data set, train data set has both input and output variable(s). ' \
           'Need to predict probability of disbursal for test data set.'

variables = """<b>About Comapny: </b>
Happy Customer Bank is a mid-sized private bank which deals in all kinds of loans. \
They have presence across all major cities in India and focus on lending products. \
They have a digital arm which sources customers from the internet.
<b>Input variables:</b>
ID - Unique ID (can not be used for predictions)
Gender- Sex
City - Current City
Monthly_Income - Monthly Income in rupees
DOB - Date of Birth
Lead_Creation_Date - Lead Created on date
Loan_Amount_Applied - Loan Amount Requested (INR)
Loan_Tenure_Applied - Loan Tenure Requested (in years)
Existing_EMI - EMI of Existing Loans (INR)
Employer_Name - Employer Name
Salary_Account- Salary account with Bank
Mobile_Verified - Mobile Verified (Y/N)
Var5- Continuous classified variable
Var1- Categorical variable with multiple levels
Loan_Amount_Submitted- Loan Amount Revised and Selected after seeing Eligibility
Loan_Tenure_Submitted- Loan Tenure Revised and Selected after seeing Eligibility (Years)
Interest_Rate- Interest Rate of Submitted Loan Amount
Processing_Fee- Processing Fee of Submitted Loan Amount (INR)
EMI_Loan_Submitted- EMI of Submitted Loan Amount (INR)
Filled_Form- Filled Application form post quote
Device_Type- Device from which application was made (Browser/ Mobile)
Var2- Categorical Variable with multiple Levels
Source- Categorical Variable with multiple Levels
Var4- Categorical Variable with multiple Level

<b>Outcomes:</b>
LoggedIn- Application Logged (Variable for understanding the problem – cannot be used in prediction)
Disbursed- Loan Disbursed (Target Variable)

<b>Problem:</b>
Digital arms of banks today face challenges with lead conversion, \
they source leads through mediums like search, display, email campaigns and via affiliate partners. \
Here Happy Customer Bank faces same challenge of low conversion ratio. \
They have given a problem to identify the customers segments having higher conversion ratio \
for a specific loan product so that they can specifically target these customers, \
here they have provided a partial data set for salaried customers only from the last 3 months. \
They also capture basic details about customers like gender, DOB, existing EMI, employer Name, \
Loan Amount Required, Monthly Income, City, Interaction data and many others. \
Let’s look at the process at Happy Customer Bank. \
In above process, customer applications can drop majorly at two stages, \
at login and approval/rejection by bank. \
Here we need to identify the segment of customers having higher disbursal rate in next 30 days \

<b>data_set</b>
We have train and test data set, train data set has both input and output variable(s).
Need to predict probability of disbursal for test data set.
"""

evaluation_criteria = 'Evaluation metrics of this challenge is ROC_AUC. ' \
                      'To read more detail about ROC_AUC refer this article “Model Evaluation Metrics 386”.'

preprocessing_descr = """<b>Dropped:</b>
<u>EmployerName</u> dropped because of too many categories
<u>ID</u> dropped - not relevant
<u>Salary_Account</u> dropped - not relevant
<u>Lead_Creation_Date</u> dropped because made little intuitive impact on outcome
<u>LoggedIn</u> dropped
<u>Salary_Account</u> dropped
<br><b>Preprocessed:</b>
<u>City</u> values changed to "S", "M", "B", "L" depending on the occurrence
<u>DOB</u> converted to Age | DOB dropped
<u>Existing_EMI</u> imputed with 0 (median) since only 111 values were missing
<u>Interest_Rate_Missing</u> created which is 1 if Interest_Rate was missing else 0
Original variable Interest_Rate dropped
<u>Loan_Amount_Applied</u>, Loan_Tenure_Applied imputed with median values
<u>EMI_Loan_Submitted_Missing</u> created which is 1 if EMI_Loan_Submitted was missing else 0
Original variable EMI_Loan_Submitted dropped
<u>Loan_Amount_Submitted_Missing</u> created which is 1 if Loan_Amount_Submitted was missing else 0
Original variable Loan_Amount_Submitted dropped
<u>Loan_Tenure_Submitted_Missing</u> created which is 1 if Loan_Tenure_Submitted was missing else 0
Original variable Loan_Tenure_Submitted dropped
<u>Processing_Fee_Missing</u> created which is 1 if Processing_Fee was missing else 0
Original variable Processing_Fee dropped
<u>Source</u> – top 2 kept as is and all others combined into different category
Numerical and One-Hot-Coding performed
"""

preprocessing_descr_list = preprocessing_descr.split('\n')
variables_list = variables.split('\n')
