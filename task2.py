import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

dataframe = pd.read_csv('./data.csv')  # Load dataset # correct location when uploading

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

cat_cols_n = [] #new category column list


# finding discrete feature attributes with max percentage larger than max_percent_val
max_percent_val = 0.6
for f in cat_cols:
    feature_ratios = yesdf[f].value_counts()/total_rows
    max_val = feature_ratios.values.max()
    if feature_ratios.max() < max_percent_val:
        del featureframe[f]
    else:
        cat_cols_n.append(f)

val_cols_n = []

# finding continuous feature attributes with max std larger than max_std

max_std = 0.25
for f in val_cols:
    norm = yesdf[f]/yesdf[f].max()
    feature_std = np.std(norm) # normalizing the dataset
    if feature_std < max_std:
        del featureframe[f]
    else:
        val_cols_n.append(f)

# selected feature set
print(' : '.join(['Selected feature set',str(featureframe.columns)]))

#Convert categorical variable into dummy/indicator variables.
ff = pd.get_dummies(featureframe, columns=cat_cols_n, drop_first=True) 

features = pd.DataFrame(ff).to_numpy()

tf = pd.get_dummies(targetframe, columns=['Churn'] , drop_first=True) 
target_name = tf.columns
target = pd.DataFrame(tf).to_numpy()
# return a flattened array
target = np.ravel(target)

#print(mutual_info_classif(features,target))

# split into training and test sets
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=25)

# using random forest
max_leaf_n = 20
estimator_number_rf = 500
rf_model = RandomForestClassifier(n_estimators=estimator_number_rf, max_leaf_nodes=max_leaf_n, n_jobs=-1)
rf_model.fit(features_train, target_train)

y_pred_rf = rf_model.predict(features_test)

print(':'.join(["random forest - precision",str(precision_score(target_test,y_pred_rf))]))
print(':'.join(["random forest - recall",str(recall_score(target_test,y_pred_rf))]))
print(':'.join(["random forest - f1-score",str(f1_score(target_test,y_pred_rf))]))

# using adaboosting
decision_tree_model = DecisionTreeClassifier(max_depth=5)

estimator_number_ada = 500
learn_rate = 1

ada_model = AdaBoostClassifier(
    base_estimator=decision_tree_model, n_estimators=estimator_number_ada,
    algorithm="SAMME.R", learning_rate= learn_rate)

ada_model.fit(features_train, target_train)

# Generate predictions for test dataset
y_pred_ada = ada_model.predict(features_test)

print(':'.join(["ada boosting - precision",str(precision_score(target_test,y_pred_ada))]))
print(':'.join(["ada boosting - recall",str(recall_score(target_test,y_pred_ada))]))
print(':'.join(["ada boosting - f1-score",str(f1_score(target_test,y_pred_ada))]))

best_model_task2 = VotingClassifier(
    estimators=[('ab', ada_model), ('rf', rf_model)],
    voting='soft')
best_model_task2.fit(features_train, target_train)

y_pred_voting = best_model_task2.predict(features_test)

print(':'.join(["voting - precision",str(precision_score(target_test,y_pred_voting))]))
print(':'.join(["voting - recall",str(recall_score(target_test,y_pred_voting))]))
print(':'.join(["voting - f1-score",str(f1_score(target_test,y_pred_voting))]))

# create confusion matrix
matrix = confusion_matrix(target_test, y_pred_voting)
class_names = ['Churn_No', 'Churn_Yes']
dataframe_Confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)

# create heatmap
sns.heatmap(dataframe_Confusion, annot=True,  cmap="Blues", fmt=".0f")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.savefig('./confusion_matrix_task_2.png')
plt.show()
plt.close()
