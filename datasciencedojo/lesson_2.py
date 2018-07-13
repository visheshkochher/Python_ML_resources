from sklearn.ensemble.forest import RandomForestClassifier
import scikitplot as skplt
from sklearn.preprocessing import Imputer, normalize
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from datasciencedojo.lesson_1 import *
from datetime import datetime
from pytz import timezone
from sklearn.linear_model import LogisticRegression


def curr_time():
    return str(datetime.now(timezone('CET')).strftime("%Y-%m-%d %H:%M:%S"))


def clean_and_engineer_df(df):

    TARGET = None
    if 'Survived' in df.columns:
        df['Survived'] = df['Survived'].astype('category')
        TARGET = df[['Survived']]

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
               'Mlle': 'Miss',
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
    df['Clean_Title'] = df.apply(
        lambda x: x['Clean_Title'] if x['Clean_Title'] != 'Dr' else 'Mr' if x['Sex'] == 'male' else 'Mrs', axis=1)

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

    df['cabin_alpha'] = df['Cabin'].map(lambda x: x.replace('\d+', '')[0] if not x.__str__() == 'nan' else "NONE")
    df['count_cabins'] = df['Cabin'].apply(lambda x: 0 if x.__str__() == 'nan' else len(x.split(' ')))
    df['has_special_cabin'] = df['cabin_alpha'].apply(lambda x: 1 if x in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] else 0)

    drop_cols = ['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId', 'Title', 'ticket_alpha', 'cabin_alpha',
                'ticket_alpha_elements']
    categorical_cols = ['Embarked', 'Clean_Title', 'Sex',
                        #'is_alone',
                        'Special_Title',
                        'has_special_cabin',
                        'is_a', 'is_paris', 'is_pc', 'is_soton', 'is_ca', 'is_ston', 'is_wep', 'is_soc', 'is_sc',
                        'is_so', 'is_pp', 'is_fcc']

    for col in categorical_cols:
        df[col] = df[col].astype('category')
    features = df.drop([col for col in drop_cols if col in df.columns], axis=1)

    ##### IMPUTE AGE BASED ON MEDIAN PER TITLE GROUP
    mean_ages = features.groupby("Clean_Title").agg({'Age': 'median'}).reset_index().rename(columns={'Age': 'AvgAge'})
    features['Age'] = features.apply(lambda x: x['Age'] if not x['Age'].__str__() == 'nan' else
    mean_ages[mean_ages['Clean_Title'] == x['Clean_Title']]['AvgAge'].values[0], axis=1)

    imp = Imputer(strategy='mean')
    features_numeric = pd.DataFrame(imp.fit_transform(features.drop(categorical_cols, axis=1)),
                                    columns=features.drop(categorical_cols, axis=1).columns)
    features_all = features_numeric
    features_dummies = features_all.join(pd.get_dummies(features[categorical_cols]))
    features_raw = features_all.join(features[categorical_cols])
    return features_dummies, TARGET, features_raw


titanic_df = pd.read_csv(
    '/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/bootcamp/Datasets/titanic.csv')
kaggle_data = pd.read_csv(
    '/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/test.csv')
all_data = titanic_df.drop(['Survived'], axis=1).append(kaggle_data, ignore_index=False)

all_data_test, _, all_data_raw = clean_and_engineer_df(all_data)


def gather_ticket_details(x):
    return pd.Series(dict(passenger_count=x['PassengerId'].count(),
                          Parch_max=x['Parch'].max(),
                          Parch_min=x['Parch'].min(),
                          SibSp_max=x['Parch'].max(),
                          SibSp_min=x['Parch'].min(),
                          names="[[%s]]" % '], ['.join(x['Name']),
                          surnames='[["%s"]]' % '"], ["'.join(x["Name"].apply(lambda x: x.split(',')[0]).astype(str)),
                          passenger_id_concat="[%s]" % ', '.join(x['PassengerId'].astype(str)),
                          min_fare=x['Fare'].min(),
                          max_fare=x['Fare'].max(),
                          min_class=x['Pclass'].min(),
                          max_class=x['Pclass'].max(),
                          total_special_title=x['Special_Title'].astype(int).sum()
                          )
                     )


ticket_data = all_data
ticket_details = ticket_data.groupby('Ticket').apply(gather_ticket_details).sort_values('passenger_count')
ticket_details1 = ticket_data.groupby('Ticket').apply(gather_ticket_details).reset_index().sort_values('Ticket')
group_surnames = ticket_details1['surnames'].apply(lambda x: eval(x)[0][0])
ticket_details1['group_surnames'] = group_surnames


def gather_surname_groups(x):
    return pd.Series(dict(passenger_count=x['passenger_count'].sum(),
                          Parch_max=x['Parch_max'].max(),
                          Parch_min=x['Parch_min'].min(),
                          SibSp_max=x['SibSp_max'].max(),
                          SibSp_min=x['SibSp_min'].min(),
                          passenger_id_concat="[%s]" % ', '.join(x['passenger_id_concat'].astype(str)),
                          ticket_concat='[["%s"]]' % '"], ["'.join(x['Ticket']),
                          min_class=x['min_class'].min(),
                          max_class=x['max_class'].max(),
                          total_special_title=x['total_special_title'].sum(),
                          # names="[[%s]]" % '], ['.join(x['Name']),
                          # surnames="[{%s}]" % ', '.join(x['Name'].apply(lambda x: x.split(',')[0]).astype(str))
                          )
                     )


consecutive_groups = ticket_details1.groupby('group_surnames').apply(gather_surname_groups).reset_index().sort_values(
    'passenger_count')


def new_relationships(df):
    row_name = df
    row_name['group_size'] = 0
    row_name['Ticket_nr'] = [(y[-1]) for y in eval(row_name['ticket_concat'])]
    row_name['Ticket_nr'] = [int(
        (y.split(' ')[-1]).replace('[[', '{').replace(']]', '}').replace('[', '').replace(']', '').replace('{',
                                                                                                           '[').replace(
            '}', ']')) for y in row_name['Ticket_nr'] if y != 'LINE']
    row_name['Ticket_raw'] = [(y[0]) for y in eval(row_name['ticket_concat'])]
    row_name['ticket_range'] = max(row_name['Ticket_nr']) - min(row_name['Ticket_nr']) if row_name['Ticket_nr'] else 0

    if any([(row_name['min_class'] - row_name['max_class'] == 0 & row_name['ticket_range'] <= 100),
            ((row_name['min_class'] - row_name['max_class'] == 0) & (
                    row_name['Parch_max'] + row_name['SibSp_max'] < row_name['passenger_count'])),
            ((row_name['ticket_range'] <= 100) & (
                    row_name['Parch_max'] + row_name['SibSp_max'] < row_name['passenger_count'])),
            (row_name['min_class'] - row_name['max_class'] == 0 & row_name['max_class'] < 3)
            ]):
        row_name['group_size'] = row_name['passenger_count']
    return row_name


consecutive_groups = consecutive_groups.apply(new_relationships, axis=1)
gathered_groups = pd.DataFrame(
    {'group_surnames': np.repeat(consecutive_groups.group_surnames.values, consecutive_groups.Ticket_raw.str.len()),
     'group_size': np.repeat(consecutive_groups.group_size.values, consecutive_groups.Ticket_raw.str.len()),
     'max_class': np.repeat(consecutive_groups.max_class.values, consecutive_groups.Ticket_raw.str.len()),
     'total_special_title': np.repeat(consecutive_groups.total_special_title.values,
                                      consecutive_groups.Ticket_raw.str.len()),
     'Ticket': np.concatenate(consecutive_groups.Ticket_raw.values)})

ticket_details1_enhanced = ticket_details1[['Ticket', 'passenger_count']].merge(gathered_groups, on='Ticket')[['Ticket','group_size', 'max_class', 'total_special_title', 'passenger_count']]
ticket_details1_enhanced['group_size'] = ticket_details1_enhanced.apply(lambda x: x['group_size'] if  x['group_size'] > 0 else  x['passenger_count'], axis = 1)
ticket_details1_enhanced.drop('passenger_count', inplace=True, axis = 1)


features_all, target, features_raw = clean_and_engineer_df(titanic_df.merge(ticket_details1_enhanced, how = 'left', on = 'Ticket'))
kaggle_data_test, _, kaggle_data_raw = clean_and_engineer_df(kaggle_data.merge(ticket_details1_enhanced, how = 'left', on = 'Ticket'))



trainX, testX, trainY, testY = train_test_split(features_all, target, test_size=0.3, random_state=297)
n_features = len(trainX.columns)
clf = RandomForestClassifier(random_state=297)
param_grid = {'max_depth': range(1, 30),
              'min_samples_split': range(2, 20),
              'min_samples_leaf': range(1, 30),
              'max_features': range(1, n_features + 1),
              'criterion': ['gini', 'entropy'],
              # 'splitter': ['best', 'random'], ### FOR DECISION TREE ONLY
              'n_estimators': range(1, 50)  ### FOR RANDOMFOREST ONLY
              }
# cv_model = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=-1)
cv_model = RandomizedSearchCV(clf, param_distributions=param_grid, cv=5, n_jobs=-1, n_iter=1000, random_state=297)
cv_model.fit(trainX, trainY.values.ravel())
cv_model.score(trainX, trainY)
cv_model.score(testX, testY)
cv_model.best_params_

logit = LogisticRegression(penalty='l2')
logit.fit((trainX), trainY.values.ravel())
logit.score((trainX), trainY)
logit.score((testX), testY)
logit.coef_
feature_importance = pd.DataFrame(logit.coef_[0], index=trainX.columns, columns=['Imp']).reset_index()
feature_importance['pk'] = 1
# plot_scatter(feature_importance, 'index', 'Imp', 'index')
plot_bar(feature_importance, 'index', 'Imp', 'index')

###CHECK THRESHOLD PERFORMANCE
predict_proba_df = pd.DataFrame(list(zip([1 - x[0] for x in cv_model.predict_proba(testX)], testY['Survived'])),
                                columns=['Predict', 'Actual'])
plot_density(predict_proba_df, 'Predict', 'Actual')
predict_proba_df['Result'] = predict_proba_df['Predict'].apply(lambda x: 1 if x >= .5 else 0)
pd.crosstab(predict_proba_df['Result'], predict_proba_df['Actual'])


### CHECK BIAS AND VARIANCE
def bias_variance_test(testX, testY, n=500):
    test_all = testX.join(testY)
    plt.close()
    i = 0
    result = []
    while i < n:
        sample_data = test_all.sample(round(len(test_all.index) * .2))
        score = cv_model.score(sample_data.drop('Survived', axis=1), sample_data['Survived'])
        result.append(score)
        print('{} %'.format(round(score * 100, 2)))
        i += 1
    plt.boxplot(result)
    plt.show()


bias_variance_test(testX, testY)

#### MAKE SUBMISSION
logit.fit(features_all, target.values.ravel())
# cv_model.best_score_
kaggle_predictions = logit.predict(kaggle_data_test)

submission = kaggle_data[['PassengerId']]
submission['Survived'] = kaggle_predictions
submission.to_csv(
    '/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/submission_{}.csv'.format(
        curr_time()), index=False)

sub1 = pd.read_csv(
    '/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/submission_2018-07-13 09:14:21.csv')
sub2 = pd.read_csv(
    '/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/submission_2018-07-11 11:06:07.csv')
sub3 = pd.read_csv(
    '/Users/visheshkochher/Desktop/Python_ML_resources/datasciencedojo/kaggle/everyone_dies/submission_2018-07-13 09:17:57.csv')

sub_all = sub3.join(sub1, lsuffix='_1', rsuffix='_2')
pd.crosstab(sub_all['Survived_1'], sub_all['Survived_2'])
cv_model.cv_results_

### ASSESS BEST PARAMS TREE AND SCORE
tree_model = RandomForestClassifier(random_state=297,
                                    **cv_model.best_params_)  ####ONLY IF THE PREVIOUS MODEL IS A SearchCV
tree_model = tree_model.fit(trainX, trainY.values.ravel())
tree_model.score(trainX, trainY)
tree_model.score(testX, testY)

### CHECK IMPORTANCE OF FEATURES
feature_importance = pd.DataFrame(tree_model.feature_importances_, index=trainX.columns, columns=['Imp']).reset_index()
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
