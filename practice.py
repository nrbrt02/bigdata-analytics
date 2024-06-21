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

titanic_df = pd.read_csv('train.csv')

#finding the median on age column
median_age = titanic_df['Age'].median()

#Replacing the where is null with median
titanic_df['Age'] = titanic_df['Age'].fillna(median_age)

#Finding all dupricated you can pass the the column name along  like titanic_df.duplicated("PassengerId") best Idea to drop on Id
duplicated = titanic_df.duplicated()

#Drop duplicated
titanic_df.drop_duplicates();

#converting the non numerical to numerical
label_encoding = preprocessing.LabelEncoder()
titanic_df["Sex"] = label_encoding.fit_transform(titanic_df['Sex'].astype(str))
titanic_df = pd.get_dummies(titanic_df, columns = ['Embarked'])

# Droping columns of no use
#Slipting data into X,Y
X = titanic_df.drop(columns=['Name', 'Ticket', 'Survived', 'Cabin', 'PassengerId'], axis=1)
Y = titanic_df["Survived"]


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
print(Y)
#Export the training training data
# X.to_csv('practice_processed.csv', index=False)
#Training the model with all the model training algorithm
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)  # Adjust hyperparameters as needed
rf_model.fit(X_train, Y_train)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, Y_train)

# Support Vector Machine (SVC)
svc_model = SVC()  # Adjust hyperparameters as needed
svc_model.fit(X_train, Y_train)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)


# Make predictions on the testing set
rf_predictions = rf_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)
svc_predictions = svc_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)

# Calculate accuracy scores
rf_accuracy = accuracy_score(Y_test, rf_predictions)
lr_accuracy = accuracy_score(Y_test, lr_predictions)
svc_accuracy = accuracy_score(Y_test, svc_predictions)
dt_accuracy = accuracy_score(Y_test, dt_predictions)


#Creating a presistance model
# best_model = RandomForestClassifier(n_estimators=100)
joblib.dump(rf_model, "bigFinal/pesistance_model.pkl")
# print(rf_predictions)
# Print the Testing accuracy scores
print("Random Forest Accuracy:", rf_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)
print("Support Vector Machine Accuracy:", svc_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)


load_joblib = joblib.load("bigFinal/pesistance_model.pkl")
# model = joblib.load("pesistance_model.pkl")  # Load the model once (assuming outside a loop)
feature_names = ["Pclass", "SibSp", "Age", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"]
new_data = np.array([[1, 0, 38, 1, 0, 71.2833, 1, 0, 0]])
# new_data = pd.DataFrame([[1, 0, 38, 1, 0, 71.2833, 1, 0, 0]], columns = feature_names)
predictions = load_joblib.predict(new_data)

print(predictions[0])

# print (duplicated)