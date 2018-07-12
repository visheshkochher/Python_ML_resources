from sklearn import tree
from sklearn.ensemble.forest import RandomForestClassifier
import scikitplot as skplt
from sklearn.preprocessing import Imputer, normalize
import pydot
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from datasciencedojo.lesson_1 import *
from datetime import datetime
from pytz import timezone
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_boston
def curr_time():
    return str(datetime.now(timezone('CET')).strftime("%Y-%m-%d %H:%M:%S"))


def clean_and_engineer_df(df):
    # df = kaggle_data
    # df = titanic_df
    TARGET = None
    if 'Survived' in df.columns:
        df['Survived'] = df['Survived'].astype('category')
        TARGET = df[['Survived']]

    # df['Pclass'] = df['Pclass'].astype('category')
    df['Embarked'] = df['Embarked'].astype('category')
    df['Sex'] = df['Sex'].astype('category')


    ### FILL NA VALUES
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].astype('category').value_counts().index[0])


    ####NEW FEATURES
    df['Title'] = df.Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False)
    df['group_size'] = df.apply(lambda x: x['Parch']+x['SibSp'], axis = 1)
    df['is_alone'] = df.apply(lambda x: 0 if x['group_size'] == 0 else 1, axis=1)
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
    special_titles = ['Lady', 'Sir', 'the Countess', 'Jonkheer', 'Capt', 'Major', 'Col', 'Don', 'Dr']
    df['Special_Title'] = [1 if x in special_titles else 0 for x in df['Title']]

    df['Clean_Title'] = df['Title']
    df['Clean_Title'] = df['Clean_Title'].map(mapping)
    df['Clean_Title'] = df.apply(lambda x: x['Clean_Title'] if x['Clean_Title'] != 'Dr' else 'Mr' if x['Sex'] == 'male' else 'Mrs', axis=1)

    df['ticket_alpha'] = df['Ticket'].apply(lambda x: x.lower().split(' ')[0])
    df['ticket_alpha'] = df['ticket_alpha'].apply(lambda x: '' if x.isnumeric() else x).str.replace('\.', '')
    ticket_alpha_mapping = {'we/p': 'wep',
                            'so/c': 'soc'}
    df['ticket_alpha_elements'] = df['ticket_alpha'].replace(ticket_alpha_mapping).apply(lambda x: x.split('/'))
    def contains_a(some_list):
        it_does = False
        for item in some_list:
            if item.startswith('a'):
                it_does = True
                break
        return it_does
    df['is_a'] = df['ticket_alpha_elements'].apply(lambda x: 1 if contains_a(x) else 0)
    df['is_paris'] = df['ticket_alpha_elements'].apply(lambda x: 1 if 'paris' in x else 0)
    df['is_pc'] = df['ticket_alpha_elements'].apply(lambda x: 1 if 'pc' in x else 0)
    df['is_soton'] = df['ticket_alpha_elements'].apply(lambda x: 1 if 'soton' in x else 0)
    df['is_ca'] = df['ticket_alpha_elements'].apply(lambda x: 1 if 'ca' in x else 0)
    df['is_ston'] = df['ticket_alpha_elements'].apply(lambda x: 1 if 'ston' in x else 0)
    df['is_wep'] = df['ticket_alpha_elements'].apply(lambda x: 1 if 'wep' in x else 0)
    df['is_soc'] = df['ticket_alpha_elements'].apply(lambda x: 1 if 'soc' in x else 0)
    df['is_sc'] = df['ticket_alpha_elements'].apply(lambda x: 1 if 'sc' in x else 0)
    df['is_so'] = df['ticket_alpha_elements'].apply(lambda x: 1 if 'so' in x else 0)
    df['is_pp'] = df['ticket_alpha_elements'].apply(lambda x: 1 if 'pp' in x else 0)
    df['is_fcc'] = df['ticket_alpha_elements'].apply(lambda x: 1 if 'fcc' in x else 0)


    df['cabin_alpha'] = df['Cabin'].map(lambda x: x.replace('\d+', '')[0] if not x.__str__()=='nan' else "NONE")
    df['count_cabins'] = df['Cabin'].apply(lambda x: 0 if x.__str__() == 'nan' else len(x.split(' ')))
    df['has_special_cabin'] = df['cabin_alpha'].apply(lambda x: 1 if x in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] else 0)

    drop_cols = ['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId', 'Title', 'ticket_alpha', 'cabin_alpha', 'group_size', 'ticket_alpha_elements']
    categorical_cols = ['Embarked', 'Clean_Title', 'Sex',
                        'is_alone',
                        'Special_Title',
                        'has_special_cabin',
                        'is_a', 'is_paris', 'is_pc', 'is_soton', 'is_ca', 'is_ston', 'is_wep', 'is_soc', 'is_sc', 'is_so', 'is_pp', 'is_fcc']

    for col in categorical_cols:
        df[col] = df[col].astype('category')
    features = df.drop([col for col in drop_cols if col in df.columns], axis=1)

    ##### IMPUTE AGE BASED ON MEDIAN PER TITLE GROUP
    mean_ages = features.groupby("Clean_Title").agg({'Age':'median'}).reset_index().rename(columns={'Age':'AvgAge'})
    features['Age'] = features.apply(lambda x: x['Age'] if not x['Age'].__str__() == 'nan' else mean_ages[mean_ages['Clean_Title'] == x['Clean_Title']]['AvgAge'].values[0], axis = 1)

    imp = Imputer(strategy='mean')
    features_numeric = pd.DataFrame(imp.fit_transform(features.drop(categorical_cols, axis=1)),
                                    columns=features.drop(categorical_cols, axis=1).columns)
    features_all = features_numeric
    features_all = features_all.join(pd.get_dummies(features[categorical_cols]))
    features_raw = features_all.join(features[categorical_cols])
    return features_all, TARGET, features_raw


titanic_df = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/bootcamp/Datasets/titanic.csv')
kaggle_data = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/test.csv')
all_data = titanic_df.drop(['Survived'], axis=1).append(kaggle_data, ignore_index=False)
features_all, target, features_raw = clean_and_engineer_df(titanic_df)
kaggle_data_test, _, kaggle_data_raw = clean_and_engineer_df(kaggle_data)


def gather_ticket_details(x):
    return pd.Series(dict(passenger_count=x['PassengerId'].count(),
                          Parch_max=x['Parch'].max(),
                          Parch_min=x['Parch'].min(),
                          SibSp_max=x['Parch'].max(),
                          SibSp_min=x['Parch'].min(),
                          names="{%s}" % ', '.join(x['Name'])))

ticket_details = all_data.groupby('Ticket').apply(gather_ticket_details).sort_values('passenger_count')

all_data.groupby('Ticket').agg({'PassengerId': 'count'})
plot_hist(all_data, 'Ticket', 'Pclass')
# plot_hist(titanic_df, 'count_cabins', 'Survived')
# plot_hist(features_raw, 'Age', 'Clean_Title')
# plot_hist(df, 'ticket_alpha', 'count_cabins')
# plot_hist(df, 'ticket_alpha', 'Survived')
# plot_hist(df, 'ticket_alpha', 'Pclass')
# plot_hist(df, 'ticket_alpha', 'Parch')
# plot_hist(df, 'ticket_alpha', 'SibSp')
# plot_density(df, 'group_size', 'Survived')
# plot_density(df, 'group_size', 'Survived')
# plot_density(df, 'is_alone', 'Survived')
# plot_hist(df, 'has_special_cabin', 'Survived')
# plot_hist(df, 'count_cabins', 'Survived')
# plot_hist(titanic_df, 'Title', 'Sex')
# plot_hist(titanic_df, 'count_cabins', 'Survived')
# plot_density(features_raw.join(target), features_raw.columns[18], 'Survived')

trainX, testX, trainY, testY = train_test_split(features_all, target, test_size=0.3, random_state=297)

# label = LabelEncoder()
# for column in categorical_cols:
#     try:
#         features_all['{}_Code'.format(column)] = label.fit_transform(features[column])
#     except TypeError:
#         print(column)


####FIT TREE WITH HYPERPARAMETER TUNING AND CROSS VALIDATION
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
cv_model = RandomizedSearchCV(clf, param_distributions=param_grid, cv=5, n_jobs=-1, n_iter=1000, random_state=297)
cv_model.fit(trainX, trainY.values.ravel())
cv_model.score(trainX, trainY)
cv_model.score(testX, testY)
cv_model.best_params_


logit = LogisticRegression(penalty='l1')
logit.fit((trainX), trainY.values.ravel())
logit.score((trainX), trainY)
logit.score((testX), testY)
logit.coef_
feature_importance = pd.DataFrame(logit.coef_[0], index = trainX.columns, columns=['Imp']).reset_index()
feature_importance['pk'] = 1
# plot_scatter(feature_importance, 'index', 'Imp', 'index')
plot_bar(feature_importance, 'index', 'Imp', 'index')


###CHECK THRESHOLD PERFORMANCE
predict_proba_df = pd.DataFrame(list(zip([1-x[0] for x in cv_model.predict_proba(testX)], testY['Survived'])), columns = ['Predict', 'Actual'])
plot_density(predict_proba_df, 'Predict', 'Actual')
predict_proba_df['Result'] = predict_proba_df['Predict'].apply(lambda x: 1 if x >= .5 else 0)
pd.crosstab(predict_proba_df['Result'], predict_proba_df['Actual'])


### CHECK BIAS AND VARIANCE
def bias_variance_test(testX, testY, n = 500):
    test_all = testX.join(testY)
    plt.close()
    i=0
    result = []
    while i<n:
        sample_data = test_all.sample(round(len(test_all.index)*.2))
        score = cv_model.score(sample_data.drop('Survived', axis=1), sample_data['Survived'])
        result.append(score)
        print('{} %'.format(round(score*100,2)))
        i += 1
    plt.boxplot(result)
    plt.show()


bias_variance_test(testX, testY)


#### MAKE SUBMISSION
cv_model.fit(features_all, target.values.ravel())
# cv_model.best_score_
kaggle_predictions = cv_model.predict(kaggle_data_test)

submission = kaggle_data[['PassengerId']]
submission['Survived'] = kaggle_predictions
submission.to_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/submission_{}.csv'.format(curr_time()), index=False)


sub1 = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/submission_2018-07-10 22:59:33.csv')
sub2 = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/submission_2018-07-11 11:06:07.csv')
sub3 = pd.read_csv('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/submission_2018-07-11 16:14:08.csv')

sub_all = sub3.join(sub1, lsuffix='_1', rsuffix='_2')
pd.crosstab(sub_all['Survived_1'], sub_all['Survived_2'])
cv_model.cv_results_


### ASSESS BEST PARAMS TREE AND SCORE
tree_model = RandomForestClassifier(random_state=297, **cv_model.best_params_) ####ONLY IF THE PREVIOUS MODEL IS A SearchCV
tree_model = tree_model.fit(trainX, trainY.values.ravel())
tree_model.score(trainX, trainY)
tree_model.score(testX, testY)

### CHECK IMPORTANCE OF FEATURES
feature_importance = pd.DataFrame(tree_model.feature_importances_, index = trainX.columns, columns=['Imp']).reset_index()
feature_importance['pk'] = 1
plot_scatter(feature_importance, 'index', 'Imp', 'index')
plot_bar(feature_importance, 'index', 'Imp', 'index')

### PREDICT
prediction = tree_model.predict(features_all)
tree_model.predict_proba(features_all)

#### VISUALIZE TREE
### ONLY FOR SIMPLE DECISION TREE
# tree.export_graphviz(tree_model,
#                      feature_names=list(trainX.columns),
#                      out_file='/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/tree.dot')
# (graph,) = pydot.graph_from_dot_file('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/tree.dot')
# graph.write_png('/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/tree.png')

skplt.metrics.plot_confusion_matrix(target, prediction, normalize=True)
pd.crosstab(target['Survived'], prediction)
sum(target['Survived'])