import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_excel('final.xlsx')

# median = df['radius_mean'].median()
df.dropna()
# print(df)
# label_encoding = preprocessing.LabelEncoder()
# df["diagnosis"] = label_encoding.fit_transform(df['diagnosis'].astype(str))


X = df.drop(columns=['id', 'diagnosis'], axis=1)
Y = df['diagnosis']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100)  # Adjust hyperparameters as needed
rf_model.fit(X_train, Y_train)

lr_model = LogisticRegression()
lr_model.fit(X_train, Y_train)

svc_model = SVC()
svc_model.fit(X_train, Y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)


rf_predictions = rf_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)
svc_predictions = svc_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)

rf_accuracy = accuracy_score(Y_test, rf_predictions)
lr_accuracy = accuracy_score(Y_test, lr_predictions)
svc_accuracy = accuracy_score(Y_test, svc_predictions)
dt_accuracy = accuracy_score(Y_test, dt_predictions)


print("Random Forest Accuracy:", rf_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)
print("Support Vector Machine Accuracy:", svc_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)


joblib.dump(rf_model, "bigFinal/final.pkl")
testing = joblib.load("bigFinal/final.pkl")


# print(df)