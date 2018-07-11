from sklearn import tree
from sklearn.ensemble.forest import RandomForestClassifier
import pandas as pd
import numpy as np
import scikitplot as skplt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer, Imputer, StandardScaler, normalize
import pydot
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from datasciencedojo.lesson_1 import *
from datetime import datetime
from pytz import timezone

def curr_time():
    return str(datetime.now(timezone('CET')).strftime("%Y-%m-%d %H:%M:%S"))


def clean_and_engineer_df(df):
    df.columns = [x.replace('-', '_').replace(' ', '') for x in df.columns]
    df['NetCapitalGain'] = df['capital_gain']-df['capital_loss']
    df['native_country'] = df['native_country'].apply(lambda x: x if x == ' United-States' else 'Foreign')

    TARGET = None
    if 'income' in df.columns:
        df['income'] = df['income'].astype('category')
        TARGET = df[['income']]

    drop_cols = ['income','capital_gain', 'capital_loss']
    categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']  # ticket_alpha
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    features = df.drop([col for col in drop_cols if col in df.columns], axis=1)


    features_all = features.drop(categorical_cols, axis=1)
    features_all = features_all.join(pd.get_dummies(features[categorical_cols]))
    features_raw = features_all.join(features[categorical_cols])
    return features_all, TARGET, features_raw


df = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/bootcamp/Datasets/AdultCensusIncome.csv')
features_all, target, features_raw = clean_and_engineer_df(df)
trainX, testX, trainY, testY = train_test_split(features_all, target, test_size=0.3, random_state=297)


n_features = len(trainX.columns)
clf = RandomForestClassifier(random_state=297)
param_grid = {'max_depth': range(1,30),
              'min_samples_split': range(2,20),
              'min_samples_leaf': range(1,30),
              'max_features': range(1,n_features+1),
              'criterion': ['gini', 'entropy'],
              #'splitter': ['best', 'random'], ### FOR DECISION TREE ONLY
              'n_estimators': range(1,50) ### FOR RANDOMFOREST ONLY
              }
# cv_model = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=-1)
cv_model = RandomizedSearchCV(clf, param_distributions=param_grid, cv=5, n_jobs=-1, n_iter=100, random_state=297)
cv_model.fit(trainX, trainY)
cv_model.score(trainX, trainY)
cv_model.score(testX, testY)



def bias_variance_test(testX, testY):
    test_all = testX.join(testY)
    plt.close()
    i=0
    result = []
    while i<500:
        sample_data = test_all.sample(round(len(test_all.index)*.2))
        score = cv_model.score(sample_data.drop('income', axis=1), sample_data['income'])
        result.append(score)
        print('{} %'.format(round(score*100,2)))
        i += 1
    plt.boxplot(result)
    plt.show()


bias_variance_test(testX, testY)
cv_model.best_params_
cv_model.best_score_
cv_model.best_estimator_
standard_deviation = cv_model.cv_results_['std_test_score'].mean()


import random
random.random.§