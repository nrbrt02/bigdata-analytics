import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_excel('data.xlsx')
# pokemon = pokemon.drop(columns=["Name","Legendary"])
x = data.drop(columns=["SNAMES ", "QUIZZES ", "ASSIGNMENTS", "MID-TERM", "FINAL", "Total Marks", "Marks /20"])
y = data["Grading "]
# # print(pokemon)
model = DecisionTreeClassifier()
model.fit(x.values, y)

prediction = model.predict([[13, 55, 33, 88]])
print(prediction)
