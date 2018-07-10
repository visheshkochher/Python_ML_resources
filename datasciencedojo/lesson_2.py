from sklearn import tree
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

# df = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/bootcamp/Datasets/titanic.csv')
# kaggle_data = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/test.csv')

def clean_and_engineer_df(df):

    TARGET = None
    if 'Survived' in df.columns:
        df['Survived'] = df['Survived'].astype('category')
        TARGET = df[['Survived']]

    # df.loc[pd.isnull(df['Embarked']), 'Embarked'] = 'U'
    df['Pclass'] = df['Pclass'].astype('category')
    df['Embarked'] = df['Embarked'].astype('category')
    df['Sex'] = df['Sex'].astype('category')


    ### FILL NA VALUES
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].astype('category').value_counts().index[0])


    ####NEW FEATURES

    df['Title'] = df.Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False)

    ### REDUCE TITLE CATEGORIES
    ## map 'Dr' based on gender
    mapping = {'Ms': 'Miss',
               'Mme': 'Miss',
               'Mlle':'Miss',
               'Lady': 'Mrs',
               'Sir': 'Mr',
               'the Countess': 'Mrs',
               'Jonkheer': 'Mr',
               'Capt': 'Mr',
               'Major': 'Mr',
               'Col': 'Mr',
               'Rev': 'Mr',
               'Don': 'Mr',
               'Mr': 'Mr',
               'Miss': 'Miss',
               'Master': 'Master',
               'Mrs': 'Mrs',
               'Dr': 'Dr'
               }
    special_titles = ['Lady', 'Sir', 'the Countess', 'Jonkheer', 'Capt', 'Major', 'Col', 'Rev', 'Don']
    df['Special_Title'] = [1 if x in special_titles else 0 for x in df['Title']]

    df['Clean_Title'] = df['Title']
    df['Clean_Title'] = df['Clean_Title'].map(mapping)
    df['Clean_Title'] = df.apply(lambda x: x['Clean_Title'] if x['Clean_Title'] != 'Dr' else 'Mr' if x['Sex'] == 'male' else 'Mrs', axis=1)

    df['ticket_alpha'] = df['Ticket'].str.replace('\d+', '').str.replace('\.', '')
    df['cabin_alpha'] = df['Cabin'].map(lambda x: x.replace('\d+', '')[0] if not x.__str__()=='nan' else "NONE")
    df['count_cabins'] = df['Cabin'].apply(lambda x: 0 if x.__str__() == 'nan' else len(x.split(' ')))

    drop_cols = ['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId', 'Title', 'ticket_alpha', 'cabin_alpha']
    categorical_cols = ['Sex', 'Embarked', 'Pclass', 'Clean_Title']  # ticket_alpha
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    features = df.drop([col for col in drop_cols if col in df.columns], axis=1)


    imp = Imputer(strategy='mean')
    features_numeric = pd.DataFrame(imp.fit_transform(features.drop(categorical_cols, axis=1)),
                                    columns=features.drop(categorical_cols, axis=1).columns)
    features_all = features_numeric
    features_all = features_all.join(pd.get_dummies(features[categorical_cols]))

    return features_all, TARGET


titanic_df = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/bootcamp/Datasets/titanic.csv')
kaggle_data = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/test.csv')


features_all, target = clean_and_engineer_df(titanic_df)
kaggle_data_test, _ = clean_and_engineer_df(kaggle_data)

# plot_hist(titanic_df, 'ticket_alpha', 'Survived')
# plot_hist(titanic_df, 'count_cabins', 'Survived')
plot_hist(features_all.join(target), 'Clean_Title', 'Survived')
# plot_hist(titanic_df, 'Title', 'Survived')
# plot_hist(titanic_df, 'Title', 'Sex')

trainX, testX, trainY, testY = train_test_split(features_all, target, test_size=0.3, random_state=297)

# label = LabelEncoder()
# for column in categorical_cols:
#     try:
#         features_all['{}_Code'.format(column)] = label.fit_transform(features[column])
#     except TypeError:
#         print(column)


####FIT TREE WITH HYPERPARAMETER TUNING AND CROSS VALIDATION
n_features = len(trainX.columns)
clf = tree.DecisionTreeClassifier(random_state=297)

param_grid = {'max_depth': range(1,30),
              'min_samples_split': range(2,20),
              'min_samples_leaf': range(1,30),
              'max_features': range(1,n_features+1),
              'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              }
# cv_model = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=-1)
cv_model = RandomizedSearchCV(clf, param_distributions=param_grid, cv=20, n_jobs=-1, n_iter=10000, random_state=297)
cv_model.fit(trainX, trainY)
cv_model.score(testX, testY)
cv_model.score(trainX, trainY)

def bias_variance_test(testX, testY):
    test_all = testX.join(testY)
    plt.close()
    i=0
    result = []
    while i<50:
        sample_data = test_all.sample(round(len(test_all.index)*.2))
        score = cv_model.score(sample_data.drop('Survived', axis=1), sample_data['Survived'])
        result.append(score)
        print('{} %'.format(round(score*100,2)))
        i+=1
    plt.boxplot(result)
    plt.show()
bias_variance_test(testX, testY)

cv_model.best_params_



#### MAKE SUBMISSION
cv_model.fit(features_all, target)
kaggle_predictions = cv_model.predict(kaggle_data_test)

submission = kaggle_data[['PassengerId']]
submission['Survived'] = kaggle_predictions


def curr_time():
    return str(datetime.now(timezone('CET')).strftime("%Y-%m-%d %H:%M:%S"))


submission.to_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/submission_{}.csv'.format(curr_time()), index=False)
# cv_model.cv_results_


### VISUALIZE TREE
tree_model = tree.DecisionTreeClassifier(random_state=297, **cv_model.best_params_) ####ONLY IF THE PREVIOUS MODEL IS A SearchCV
tree_model = tree_model.fit(trainX, trainY)
tree.export_graphviz(tree_model,
                     feature_names=list(trainX.columns),
                     out_file='/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/tree.dot')
(graph,) = pydot.graph_from_dot_file('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/tree.dot')
graph.write_png('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/tree.png')

### PREDICT AND SCORE
prediction = tree_model.predict(features_all)
tree_model.predict_proba(features_all)
tree_model.score(trainX, trainY)
tree_model.score(testX, testY)
tree_model.feature_importances_



skplt.metrics.plot_confusion_matrix(target, prediction, normalize=True)
pd.crosstab(target['Survived'], prediction)
sum(target['Survived'])

from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA

pca = PCA(n_components=20)
pca.fit(trainX)
trans_data = pca.transform(trainX)

svc = SVC(C=.5)
svc.fit(trans_data, trainY)
svc.score(trans_data, trainY)
svc.score(pca.transform(testX), testY)

### PREDICT AND SCORE
prediction = svc.predict(pca.transform(testX))
svc.predict_proba(features_all)
svc.score(trainX, trainY)
svc.score(testX, testY)
svc.feature_importances_


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit()