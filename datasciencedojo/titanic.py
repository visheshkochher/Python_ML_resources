import pandas as pd
from datasciencedojo import lesson_1
df = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/bootcamp/Datasets/titanic.csv')
df['Survived'] = df['Survived'].astype('category')
df['Survived'] = df['Survived'].map({0:'Dead', 1:'Survived'})
df['Embarked'] = df['Embarked'].map({'':'Unknown', 'C':'Cherbourg', 'Q':'Queenstown', 'S': 'Southampton'})
# df['Survived'].value_counts().reset_index()
# df.describe()
# df.groupby('Ticket').agg({'Survived':'count'}).reset_index().sort_values('Survived')


lesson_1.plot_density(df[['Age', 'Survived']].dropna(axis = 0), 'Age', 'Survived')