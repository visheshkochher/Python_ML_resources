import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import
from datasciencedojo.lesson_1 import *
data = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/real_estate.csv')
target_col = 'SalePrice'
drop_cols = ['Id']+[target_col]
cat_cols = []
for i in range(len(data.columns)):
    if data.dtypes[i].__str__() == 'object':
        cat_cols.append(data.columns[i])

features_all = data.drop(cat_cols, axis = 1).join(pd.get_dummies(data[cat_cols]))
features_all.dropna(inplace=True)

trainX, testX, trainY, testY = train_test_split(features_all.drop(drop_cols, axis = 1), features_all[target_col], test_size=0.3, random_state=297)
reg = LinearRegression(normalize=True)
reg.fit(trainX, trainY)
reg.score(testX, testY)
reg.score(trainX, trainY)


coef = reg.coef_
intercept = reg.intercept_
coef_result = pd.DataFrame(coef.flatten(), columns=['coef']).join(pd.DataFrame(trainX.columns, columns=['col']))
coef_result['pk'] = 1
plot_bar(coef_result, 'col', 'coef', 'pk')