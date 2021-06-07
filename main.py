import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import normalize

'''
1. categorial
    a) nominal
    b) ordered
2. Numeryczne
'''


def main():
    data = pd.read_csv('data/Train_nyOWmfK.csv', encoding="latin1")
    # print(data.shape)
    # print(data.info())
    
    #
    # # getting rid of irrelevant features
    #
    irrelevant_features = ["DOB", "Lead_Creation_Date", "ID", "Employer_Name","Salary_Account"]
    data.drop(irrelevant_features, axis=1, inplace=True)
    
    #
    # # Dealing with categorical data
    #
    
    # # Gender (feature)
    data["Gender"] = data.Gender.map({"Female": 1, "Male": 0})
    # checking the proportions of gender values
    # print(data.Gender.value_counts())

    
    # # City (feature)
    # Splitting data into categories
    data['City'].replace(dict(pd.cut(data.City.value_counts(),
                                     right=True,
                                     bins=[0, 100, 1000, 5000, 99999],
                                     labels=['S', 'M', 'B', 'L'])),
                         inplace=True)
    
    # replaces NaN values with random not NaN values in City feature
    data["City"][np.array(data.City.isna())] = np.array(data.City[~data.City.isna()].sample(data.City.isna().sum()))

    # # Mobile_Verified (feature)
    data["Mobile_Verified"] = (data.Mobile_Verified == "Y").astype(int)


    # # Source (feature)
    data["Source"] = data.Source.apply(lambda x: x if x in ["S122", "S133"] else "Other")


    # # Filled_Form (feature)
    data["Filled_Form"] = (data.Filled_Form == "Y").astype(int)


    # # Device_Type (feature)
    data["Device_Type"] = (data.Device_Type == "Mobile").astype(int)


    # # Monthly income (feature)
    
    # checking outliers
    # px.line(data.Monthly_Income.quantile(np.arange(0,1,0.01))).show('browser')  # based on plot we take 95 percentile
    
    # changing outliers to the value of 95th percentile
    mask = data.Monthly_Income > data.Monthly_Income.quantile(0.95)
    data.Monthly_Income[mask] = data.Monthly_Income.quantile(0.95)
    # px.line(data.Monthly_Income.quantile(np.arange(0,1,0.01))).show('browser')  # checking if above method worked
    
    
    # TODO check if this variables are important
    # # Loan_Amount_Applied
    # getting rid of variables (rows) that consists of nan in this feature
    data.dropna(subset=["Loan_Amount_Applied"], inplace=True)
    # # Loan_Tenure_Applied
    
    # # Loan_Amount_Submitted
    # # Loan_Tenure_Submitted
    
    
    
    #
    # # Dealing with datetime data - got rid of this data as irrelevant for the predictions
    #
    # # DOB
    # # Lead_Creation_Date
    
    #
    # # Dealing with nan values
    #
    # TODO is it a correct idea to get rid of nan?
    data.fillna(-1, inplace=True)



    #
    # # Dealing with numerical data, normalization
    #
    # WHAT a lot of values to normalize is set to '-1' which gives negative results after normalization
    to_normalize = ['Loan_Amount_Submitted','Loan_Tenure_Submitted','Loan_Amount_Applied','Loan_Tenure_Applied', 'Var5']
    data = pd.concat((data.drop(to_normalize,axis=1),
                      pd.DataFrame(normalize(data.loc[:,to_normalize]), columns=to_normalize)),
                      axis=0)
    
    print(data.Loan_Tenure_Submitted.describe())
    #
    # # One Hot Encoding data
    #
    to_one_hot = ['Var1', 'City', 'Var2','Var4', 'Source']

    X = data.drop(["LoggedIn", "Disbursed"], axis=1)
    X = pd.get_dummies(X,columns=to_one_hot)
    y = data.Disbursed

    #
    # # Machine learning part
    #


    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from xgboost.sklearn import XGBClassifier

    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=15000)
    # y_train.mean(), y_test.mean()

    # for model in [DecisionTreeClassifier(min_samples_leaf=1),
    #               RandomForestClassifier(200, min_samples_leaf=1),
    #               XGBClassifier(n_estimators=50, min_child_weight=1)]:
    #
    #     model.fit(X_train, y_train)
    #     print(f1_score(y_test, model.predict(X_test)))
    #
    # for model in [XGBClassifier(n_estimators=20),
    #               XGBClassifier(n_estimators=100)]:
    #
    #     model.fit(X_train, y_train)
    #     print(f1_score(y_test, model.predict_proba(X_test)[:,1]>0.1))
    #
    # (model.predict_proba(X_test)[:,1]>0.3).sum()
    #
    # model = DecisionTreeClassifier(class_weight={0:1, 1:100})
    # model.fit(X_train, y_train)
    # print(f1_score(y_test, model.predict(X_test)))
    #
    # model.predict(X_test).sum()
    #
    #
    # model = RandomForestClassifier(class_weight={0:1, 1:100})
    # model.fit(X_train, y_train)
    # print(f1_score(y_test, model.predict(X_test)))
    #
    #
    #
    #
    #
    
    return X,y


if __name__ == "__main__":
    X, y = main()
