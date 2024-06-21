import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('General QuiZ.csv')
# above_30 = df[df["MID-TERM"] < 30]

df.fillna(df.mode()) # 1.a

df = df.drop(columns=['SNAMES ']) #1.c
df['QUIZZES '].count() #2.a
df['MID-TERM'].mode() #2.b
print(above_30)






























# df_xlsx = pd.read_excel('pokemon_data.xlsx')
# print(df.tail(4))
# df_txt = pd.read_csv('pokemon_data.txt', delimiter='\t')

# print(df.columns)
# print(df[['Name', 'Type 1', 'HP']])

# Loop through all the rows
# for index, row in df.iterrows():
#     print(index, row['Name'])

# print (df.loc[df['Type 1'] == "Fire"])
# print(df.sort_values('Type 1', ascending=False))


#Making Changes

# df['Total'] = df['HP'] + df['Attack'] + df['Defense']
# df['Total'] = df.iloc[:, 4:10].sum(axis=1)
# print(df.head(5))

#changing the column position
# df['Total'] = df.iloc[:, 4:9]
# cols = list(df.columns)
# df = df[cols[0:4] + [cols[-1]] + cols[4:11]]



# df.plot(kind='scatter ', x='QUIZZES ', y='MID-TERM')
# x_data = df['QUIZZES ']
# y_data = df['MID-TERM']
# plt.scatter(x_data, y_data)
# plt.show()
# print(df.head())
# print(df.mode())
