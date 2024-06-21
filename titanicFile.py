import sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score




titanic_df = pd.read_csv('train.csv')
# print(titanic_df.head(10))
# print(titanic_df.shape)

#Droping unneccessly columns
titanic_df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

#Checking for missing or null value
#null or na => all means thesame
#titanic_df[titanic_df.isnull().any(axis=1)].count()

#droping all the records with a missing value
titanic_df = titanic_df.dropna()

#Statistical descriptionst
# desc = titanic_df.describe()

# finding a row based in a single column
# max_age = titanic_df["Age"].max()
# max_age_row = titanic_df.loc[titanic_df["Age"] == max_age]
# print(max_age_row)


#Ploting
# plt.scatter(titanic_df['Age'], titanic_df['Survived'])
# plt.xlabel("Age")
# plt.ylabel("Survived")
# plt.show()

# Making a matrix to se the relationship between columns
# pd.crosstab(titanic_df['Sex'], titanic_df['Survived'])
# print(pd.crosstab(titanic_df['Pclass'], titanic_df['Survived']))

#finding the total number of Female in the dataset
# all_female = titanic_df[titanic_df["Sex"] == "female"].shape[0]
# print(all_female)

# printing the correrration !!!
# titanic_df_corr = titanic_df.corr()
#Heatmap
# sns.heatmap(titanic_df_corr, annot=True)
# print(titanic_df_corr)

#Encoding the some columns(converting them to numeric value to be use in sklearn)
label_encoding = preprocessing.LabelEncoder()
titanic_df["Sex"] = label_encoding.fit_transform(titanic_df['Sex'].astype(str))
#Converting the Embarked column int a numeric value Second tech
titanic_df = pd.get_dummies(titanic_df, columns = ['Embarked'])
# titanic_df["Embarked"] = label_encoding.fit_transform(titanic_df['Embarked'].astype(str))
#Geting the Original converted values
# label_encoding.classes_

#shuffering data for better ML(helps to avoid ML to learn from sorted data)
titanic_df = titanic_df.sample(frac=1).reset_index(drop=True)

#Output the suffered dat in an csv file without the indexes
titanic_df.to_csv('titanic_processed.csv', index=False)

#Reopening the proccessed file for prediction
titanic_df = pd.read_csv("titanic_processed.csv")

#Droping the Survived cullumn and storing the remaining columns in a new valiable X
X = titanic_df.drop(columns=["Survived"], axis=1)

#Y stands for the value to be predicted which is Survival
Y = titanic_df["Survived"]

#train_test_slip automatical shuffer the data no need to do it like wee did above
# test_size=0.2 Indicates that we are saving 20% of our data for testing purpose
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

#Penalty helps us to train our model on complex models(Regularization)
#C=1.0 specify the strength of the Regularization(the smaller the strongest)
# solver = "liblinear" Algorithm user to optimize the ploblem(liblinear works well on small datasets)
#This starts the training of the the prediction model
logistic_model = LogisticRegression(penalty='l2', C=1.0, solver = "liblinear").fit(x_train, y_train)

#finaly twe predict on the remaining data in the dataset
y_pred = logistic_model.predict(x_test)

# pred_test = pd.DataFrame({'y_test': y_test,
#                           'y_pred': y_pred})

#Culculcating the Accuracy, Precision and Recall for our model
# You should have atleast accuracy of above 50%
#Accuracy is how many predicted values did the model get right
#Precision is How many did the model get write which is actually right
# Recall is how many of the actual survivors did the model correctly predict
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Acc:", acc)
print("Prec:", prec)
print("Recall:", recall)
