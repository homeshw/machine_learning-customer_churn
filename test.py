import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#from sklearn.tree import export_graphviz
#import os

dataframe = pd.read_csv('data.csv')  # Load dataset # correct location when uploading

dataframe.dropna(inplace = True) # to drop any not available values

featureframe = dataframe.iloc[:, 1 : -1 ] # filtering out the customerid and churn columns
targetframe = dataframe.iloc[:,  -1 ] # taking only the churn column

cat_cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService',
'MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
'PaymentMethod'] 

val_cols = ['tenure','MonthlyCharges','TotalCharges']

# finding features which are more suitable
yesdf = dataframe.loc[dataframe['Churn'] == 'Yes']


total_rows = yesdf['customerID'].count()

cat_cols_n = []

temp_cat_cols = cat_cols
# finding each features attribute with max percentage
for f in temp_cat_cols:    
    feature_ratios = yesdf[str(f)].value_counts()/total_rows
    print(feature_ratios)

max_std = 0.25
for f in val_cols:
    norm = yesdf[f]/yesdf[f].max()
    feature_std = np.std(norm) # normalizing the dataset
    print(f)
    print(feature_std)

#print(featureframe.columns)
#print(cat_cols)