import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/real_estate.csv')
target = data['SalePrice']
data.dtypes[0].__str__()
drop_cols = ['Id', 'SalePrice']

cat_cols = []
for i in range(len(data.columns)):
    if data.dtypes[i].__str__() == 'object':
        cat_cols.append(data.columns[i])

features_all = data.drop(cat_cols, axis = 1).join(pd.get_dummies(data[cat_cols]))
features_all.dropna(inplace=True)

trainX, testX, trainY, testY = train_test_split(features_all.drop(drop_cols, axis = 1), features_all['SalePrice'], test_size=0.3, random_state=297)
reg = LinearRegression()
reg.fit(trainX, trainY)
reg.score(testX, testY)
reg.score(trainX, trainY)
