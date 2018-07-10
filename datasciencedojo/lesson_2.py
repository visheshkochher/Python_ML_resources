from sklearn import tree
import pandas as pd
import scikitplot as skplt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer, Imputer, StandardScaler, normalize
import pydot
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline

df = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/bootcamp/Datasets/titanic.csv')
df.loc[pd.isnull(df['Embarked']), 'Embarked'] = 'U'
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df['Sex'] = df['Sex'].astype('category')

drop_cols = ['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId']
categorical_cols = ['Sex', 'Embarked', 'Pclass']

features = df.drop(drop_cols, axis=1)
target = df[['Survived']]

imp = Imputer(strategy='mean')
features_numeric = pd.DataFrame(imp.fit_transform(features.drop(categorical_cols, axis=1)),
                                columns=features.drop(categorical_cols, axis=1).columns)
features_all = features_numeric
features_all = features_all.join(pd.get_dummies(features[categorical_cols]))

# features_all = pd.DataFrame(normalize(features_all), columns=features_all.columns)
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

param_grid = {'max_depth': range(1,100),
              'min_samples_split': range(2,50),
              'min_samples_leaf': range(1,50),
              'max_features': range(1,n_features+1),
              'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              }
# cv_model = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=-1)
cv_model = RandomizedSearchCV(clf, param_distributions=param_grid, cv=100, n_jobs=-1, n_iter=10000, random_state=297)
cv_model.fit(trainX, trainY)
cv_model.score(testX, testY)
cv_model.score(trainX, trainY)

cv_model.best_params_
cv_model.cv_results_


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