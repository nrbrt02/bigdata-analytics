import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv(r"diabetes.csv")
# print(df.to_string())

# # Checking for missing values
# print(df.isnull().sum())

# Correlation
print("-----------Correlation between different columns--------")
print(df.corr())
dia = sns.heatmap(df.corr())
# plt.show()


X = df.drop("Outcome", axis=1)
Y = df["Outcome"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

Decision_tree_model= DecisionTreeClassifier()
Logistic_regression_Model=LogisticRegression(solver='lbfgs',max_iter=10000)
SVM_model=svm.SVC(kernel='linear')
RF_model=RandomForestClassifier(n_estimators=100)

##TRAINING THE MODELS
Decision_tree_model.fit(X_train, Y_train)
Logistic_regression_Model.fit(X_train, Y_train)
SVM_model.fit(X_train, Y_train)
RF_model.fit(X_train, Y_train)

##Predictiong
DT_Prediction =Decision_tree_model.predict(X_test)
LR_Prediction =Logistic_regression_Model.predict(X_test)
SVM_Prediction =SVM_model.predict(X_test)
RF_Prediction =RF_model.predict(X_test)

##MODEL ACCURACY
DT_score=accuracy_score(Y_test, DT_Prediction)
lR_score=accuracy_score(Y_test, LR_Prediction)
SVM_score=accuracy_score(Y_test, SVM_Prediction)
RF_score=accuracy_score(Y_test, RF_Prediction)

print(LR_Prediction)
print("Decision tree classifier accuracy score ",DT_score*100,"%")
print("Logistic Regression accuracy score ",lR_score*100,"%")
print("SVM accuracy score ",SVM_score*100,"%")
print("Random forest accuracy score ",RF_score*100,"%")