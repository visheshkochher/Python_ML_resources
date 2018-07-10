from sklearn import tree
import pandas as pd
import scikitplot as skplt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer, Imputer
import pydot
from sklearn.model_selection import train_test_split, GridSearchCV


df = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/bootcamp/Datasets/titanic.csv')

drop_cols = ['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId']
categorical_cols = ['Sex', 'Embarked', 'Pclass']

df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
df.loc[pd.isnull(df['Embarked']), 'Embarked'] = 'U'
df['Embarked'] = df['Embarked'].astype('category')
df['Sex'] = df['Sex'].astype('category')

features = df.drop(drop_cols, axis=1)
target = df[['Survived']]

imp = Imputer(strategy='mean')
features_numeric = pd.DataFrame(imp.fit_transform(features.drop(categorical_cols, axis = 1)), columns=features.drop(categorical_cols, axis = 1).columns)
features_all = features_numeric
features_all = features_all.join(pd.get_dummies(features[categorical_cols]))


trainX, testX, trainY, testY = train_test_split(features_all, target, test_size=0.3)

# label = LabelEncoder()
# for column in categorical_cols:
#     try:
#         features_all['{}_Code'.format(column)] = label.fit_transform(features[column])
#     except TypeError:
#         print(column)


####FIT TREE WITH HYPERPARAMETER TUNING AND CROSS VALIDATION
n_features = len(trainX.columns)
clf = tree.DecisionTreeClassifier()

param_grid = {'max_depth': range(1,11),
              'min_samples_split': range(2,11),
              'min_samples_leaf': range(1,11),
              'max_features': range(1,n_features+1),
              'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              }
cv_model = GridSearchCV(clf, param_grid=param_grid, cv=5)
# clf = clf.fit(trainX, trainY)
cv_model.fit(trainX, trainY)



### VISUALIZE TREE
tree.export_graphviz(clf, feature_names=list(features_all.columns), out_file='/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/tree.dot')
(graph,) = pydot.graph_from_dot_file('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/tree.dot')
graph.write_png('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/tree.png')

### PREDICT AND SCORE
prediction = clf.predict(features_all)
clf.predict_proba(features_all)
pd.DataFrame(clf.decision_path(features_all).to_array())
clf.score(trainX, trainY)
clf.score(testX, testY)
clf.feature_importances_


skplt.metrics.plot_confusion_matrix(target, prediction, normalize=True)
pd.crosstab(target['Survived'], prediction)
sum(target['Survived'])