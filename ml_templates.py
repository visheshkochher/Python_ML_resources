'''
###CLASSIFICATION
'''

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.forest import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(iris.data, iris.target)
knn.predict(iris.data)

len(iris.target)
sum(iris.target == knn.predict(iris.data))
knn.score(iris.data, iris.target)
help(cross_val_predict)
cross_val_predict(knn, iris.data, iris.target, cv=20)
cross_val_score(knn, iris.data, iris.target, cv=20).mean()


rf = RandomForestClassifier(n_estimators=3)
rf.fit(iris.data, iris.target)
rf.predict_proba(iris.data)
rf.score(iris.data, iris.target)


from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
'''
https://github.com/drivendataorg/box-plots-sklearn/blob/master/src/data/multilabel.py
'''
mcr = OneVsRestClassifier(LogisticRegression())
mcr.fit(iris.data, iris.target)
mcr.predict(iris.data)
mcr.predict_proba(iris.data)


'''
# Import Lasso
'''
from sklearn.linear_model import Lasso ## OR Ridge

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.2, normalize=True)
iris = datasets.load_boston()
# Fit the regressor to the data
lasso.fit(iris.data, iris.target)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
lasso.score(iris.data, iris.target)
# Plot the coefficients

# import seaborn as sns
# sns.set()
# _ = plt.hist(iris.data[5])
# _ = plt.plot(range(len(iris.feature_names)), list(zip(lasso_coef+3, lasso_coef, lasso_coef+1)))
# _ = plt.xticks(range(len(iris.feature_names)), iris.feature_names, rotation=60)
# _ = plt.margins(0.02)
# plt.show()
# plt.close()
# plt.
#



##############################
##############################
##########   SVM   ###########
##############################
##############################
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

from sklearn.linear_model import LogisticRegression

# log_reg = LogisticRegression(penalty='l1')#C=1000,
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs') ######### MULTINOMIAL CLASSIFICATION ################
log_reg.fit(X_train, y_train)
log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)


coef = log_reg.coef_
intercept = log_reg.intercept_
np.argsort(coef.flatten())[::-1]
pd.DataFrame(coef.flatten()).plot()
logreg_compare = pd.DataFrame(log_reg.predict_proba(X_test)).join(pd.DataFrame(y_test, columns=['actual']))

log_reg.coef_
log_reg.intercept_
test_1 = X_test[0]
###CALCULATE RAW MODEL OUTPUT
log_reg.coef_@test_1+log_reg.intercept_
log_reg.predict_proba(X_test)[0]
y_test[0]



##### SCIKIT PLOT #####
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test, log_reg.predict(X_test), normalize=True)
skplt.metrics.plot_precision_recall(y_test, log_reg.predict_proba(X_test))







##### PLOTTIMG DECISION REGIONS ######
# Plotting decision regions
model_1 = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_2 = LinearSVC()
model_1.fit(X_train[:,:2], y_train)
model_2.fit(X_train[:,:2], y_train)

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(([[0, 0], [0, 1]]),
                        [model_1, model_2],
                        ['Logit Multinomial', 'SVC']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()

'''
######PLOTLY TO EXPLORE DATA
'''
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Target'] = iris.target

'''Histogram
#Multiple histograms
'''
df.hist()
# OR
data = [go.Histogram(x=df.Target, histnorm='probability',
                     cumulative=dict(enabled=True)),
        go.Histogram(x=df.Target + 5, histnorm='probability')]
layout = go.Layout(barmode='stacked'  # 'overlay'
                   )
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig)

'''
###Scatter/Line
'''
traces = [
    go.Scatter(x=df.PTRATIO, y=df.Target,  # z=df.CRIM,
               #               mode='lines+markers',
               name='Name',
               text='TEXT',
               # hoverinfo = 'x+text',
               hoveron='points',
               y0=0,
               # line = dict(color='rgb(66, 196, 247)',
               #             width=4,
               #             dash='dash'
               #             ),
               marker=dict(
                   size=df.RM,
                   colorscale='Virdis',
                   line=dict(
                       width=2,
                       color='rgb(0, 0, 0)'
                   )
               )
               ),
    go.Scatter(x=df.PTRATIO, y=df.Target + 10,  # z=df.CRIM,
               #               mode='lines+markers',
               name='Name',
               text='TEXT',
               # hoverinfo = 'x+text',
               hoveron='points',
               y0=0,

               # line = dict(color='rgb(66, 196, 247)',
               #             width=4,
               #             dash='dash'
               #             ),
               marker=dict(
                   size=df.RM,
                   colorscale='Virdis',
                   line=dict(
                       width=2,
                       color='rgb(0, 0, 0)'
                   )
               )
               )
]
plot_layout = go.Layout(
    hovermode='closest',
    legend=dict(orientation="h"),
    # title='{} per {} {}'.format(rename_column_dict.get(data_content, data_content),
    #                             date_grouping_dict[time_agg_dropdown],
    #                             ('per ' + ' per '.join(
    #                                 [rename_column_dict.get(x, x) for x in
    #                                  group]) if group != [] else '')
    #                             ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis={
        "range": [
            min(df.PTRATIO),
            max(df.PTRATIO)
        ],
        # "type": "date",
        "autorange": True,
        "title": "PTRATIO"
    }
)

_ = plotly.offline.plot({"data": traces, "layout": plot_layout})
'''
###Heatmap
# go.Heatmap
'''
trace = [{'x': [1, 2, 3, 4, 5],
          'y': [1, 2, 3, 4, 5],
          'z': [1, 2, 3, 4, 5],
          'name': 'trace1',
          # 'mode': 'lines+markers',
          'type': 'scatter3d'}]
layout = {
    "autosize": True
}
fig = go.Figure(data=trace, layout=layout)
plotly.offline.plot(fig)




#####################################
#####################################
#####################################
#####################################
from sklearn.metrics import mean_squared_error
help(mean_squared_error)

from scipy.stats import randint
randint(1, 9)

from scipy.stats import pearsonr
correlation, pvalue = np.pearsonr()
#####################################
#####################################
#####################################
#####################################








#####################################
#####################################
#######UNSUPERVISED LEARNING#########
#####################################
#####################################
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import plotly
import plotly.graph_objs as go

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_target = iris.target
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)
labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

data = [go.Scatter(x=df.iloc[:, 0],
                   y=df.iloc[:, 2],
                   mode='markers',
                   marker=dict(
                       size='16',
                       color=iris.target,
                       symbol=labels,
                       # colorscale='Virdis',
                       showscale=True,
                       opacity=0.5
                   ),
                   ),
        go.Scatter(x=centroids[:, 0],
                   y=centroids[:, 2],
                   mode='markers',
                   marker=dict(
                       size='26',
                       color='yellow',
                       symbol='119'
                   ),
                   )
        ]
plotly.offline.plot(data)

###EVALUATING A CLUSTER#####

cross_tab = pd.crosstab(labels, iris.target)

inertia_list = []
for k in range(1, 8):
    model = KMeans(n_clusters=k)
    model.fit(df)
    inertia_list.append(model.inertia_)
inertia_data = [go.Scatter(x=list(range(1, 8)),
                           y=inertia_list,
                           mode='lines+markers')]
plotly.offline.plot(inertia_data)

###PIPELINE AND SCALING###
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.cluster import KMeans

scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler, kmeans)
label = pipeline.fit_predict(df)
pd.crosstab(labels, df_target)

#####################################################
###########HIERARCHICAL CLUSTERING###################
#####################################################

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

eurovision = pd.DataFrame(iris.data, columns=iris.feature_names)

eurovision['Target'] = iris.target
eurovision.drop('Target', axis=1)
mergings = linkage(iris.data, method='complete')  # method='single'
dendrogram(mergings,
           labels=iris.target,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()

from scipy.cluster.hierarchy import fcluster

labels = fcluster(mergings, 3.5, criterion='distance')

pd.crosstab(labels, iris.target)

###################################
###################################
############## t-SNE ##############
###################################
###################################
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
imp = Imputer()
df = imp.fit_transform(df)
model = TSNE(learning_rate=100)
transformed = model.fit_transform(df)

#####################################################
###########            PCA        ###################
#####################################################
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
import plotly.graph_objs as go
import plotly

pca = PCA(n_components=2)

pca.fit(iris.data)
trans_data = pca.transform(iris.data)
pca.components_
plotly.offline.plot([
    go.Scatter(x=iris.data[:, 0],
               y=iris.data[:, 2],

               mode='markers',
               marker=dict(size='16',
                           color=iris.target,
                           showscale=True,
                           opacity=0.5,
                           colorscale='Jet'
                           )
               ),
    go.Scatter(x=trans_data[:, 0],
               y=trans_data[:, 2],
               mode='markers',
               marker=dict(size='16',
                           color=iris.target,
                           showscale=True,
                           opacity=0.5,
                           colorscale='Jet',
                           symbol = '121'
                           ))
])

###See intrinsic dimensions
features = list(range(pca.n_components_))
variance = pca.explained_variance_
plotly.offline.plot([go.Bar(x=features, y=variance)])
# Get the mean of the iris samples: mean
mean = pca.mean_
# Get the first principal component: first_pc
first_pc = pca.components_[0,:]



#####################################################
###########            NMF        ###################
#####################################################
from sklearn.decomposition import NMF

model = NMF(n_components=2)
model.fit(iris.data)
features = model.transform(iris.data)

plotly.offline.plot([
    go.Scatter(x=iris.data[:, 0],
               y=iris.data[:, 2],

               mode='markers',
               marker=dict(size='16',
                           color=iris.target,
                           showscale=True,
                           opacity=0.5,
                           colorscale='Jet'
                           )
               ),
    go.Scatter(x=features[:, 0],
               y=features[:, 2],
               mode='markers',
               marker=dict(size='16',
                           color=iris.target,
                           showscale=True,
                           opacity=0.5,
                           colorscale='Jet',
                           symbol = '121'
                           ))
])

###See intrinsic dimensions and reconstruct original dataset
components = model.components_
np.matmul(features[0], components) #OR features[0].dot(components)
iris.data[0,:]


###CALCULATING COSINE SIMILARITIES TO BUILD RECOMMENDATION ENGINES
from sklearn.preprocessing import normalize

normalized_features = normalize(features)
df = pd.DataFrame(normalized_features, index=iris.target)
current_row = df.iloc[-1]
similarities = df.dot(current_row) ###COSINE SIMILARITIES
similarities.nlargest()
most_relevant = [True if i > .9 else False for i in similarities]
recommendation = df.loc[most_relevant].index






####################################################
####################################################
####################################################

###############NETWORK ANALYSIS#####################

####################################################
####################################################
####################################################
import networkx as nx
G=nx.Graph()
G.add_nodes_from([1,2,3,5])

G.add_edges_from([(2,3),(2,1), (2,5)])


G.node[1]['date']='2018-06-10'
G.node[2]['date']='2018-06-11'
G.node[3]['date']='2018-06-12'
G.node[5]['date']='2018-06-15'

# G.edge[1]['date']='2018-06-10'
# G.node[2]['date']='2018-06-11'
# G.node[3]['date']='2018-06-12'
# G.node[5]['date']='2018-06-15'



G.nodes(data=True)
nx.draw(G)
G.edges(data=True)
G.has_edge(1,2)
(1,2) in G.edges()

for n, d in G.nodes(data=True):
    # Calculate the degree of each node: G.node[n]['degree']
    G.node[n]['degree'] = nx.degree(G)[n]

# Create the ArcPlot object: a
a = nv.ArcPlot(G, node_order='degree', node_labels=True)
a.draw()
import nxviz as nv
import matplotlib.pyplot as plt
ap = nv.ArcPlot(G, node_order='date', node_color='date', node_labels=True, node_size='date')
ap.draw()

h = nv.MatrixPlot(G)#, node_grouping='grouping')
h.draw()

c = nv.CircosPlot(G, node_order='degree', node_grouping = 'date', node_color='date')
c.draw()
###IDENTIFYING IMPORTANT NODES
G.neighbors(1)
nx.degree_centrality(G)
nx.betweenness_centrality(G)

# Define path_exists()
def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    queue = [node1]

    for node in queue:
        neighbors = G.neighbors(node)
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break

        else:
            visited_nodes.add(node)
            queue.extend([n for n in neighbors if n not in visited_nodes])

        # Check to see if the final element of the queue has been reached
        if node == queue[-1]:
            print('Path does not exist between nodes {0} and {1}'.format(node1, node2))

            # Place the appropriate return statement
            return False

path_exists(G, 1, 2)


#######CLIQUES########
from itertools import combinations


# Define is_in_triangle()
def is_in_triangle(G, n):
    """
    Checks whether a node `n` in graph `G` is in a triangle relationship or not.

    Returns a boolean.
    """
    in_triangle = False

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):

        # Check if an edge exists between n1 and n2
        if G.has_edge(n1, n2):
            in_triangle = True
            break
    return in_triangle


def node_in_open_triangle(G, n):
    """
    Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`.
    """
    in_open_triangle = False

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):

        # Check if n1 and n2 do NOT have an edge between them
        if not G.has_edge(n1, n2):
            in_open_triangle = True

            break

    return in_open_triangle


list(nx.find_cliques(G))



####### SUB GRAPH  for node 8 neighbors#######
sim_G = nx.erdos_renyi_graph(n=20, p=.2)
nodes = sim_G.neighbors(8)
nodes.append(8)

G_eight = sim_G.subgraph(nodes)
nx.draw(G_eight, with_labels=True)
list(nx.find_cliques(G_eight))

sim_G.add_node(20)
connected_subgraph = list(nx.connected_component_subgraphs(sim_G))
[i.nodes() for i in connected_subgraph]
nx.draw(sim_G)
G.nodes(data=True)





from collections import defaultdict

# Initialize the defaultdict: recommended
recommended = defaultdict(int)
recommended[('a','b')] += 1
recommended[('b','a')] += 1
recommended[('c','b')] += 1
recommended.items()
from sklearn.model_selection import






######### LOG LOSS ##############
'''PENALIZES WRONG PREDICTIONS

IT IS BETTER TO BE LESS CONFIDENT THAN CONFIDENT AND WRONG'''

np.clip
error_for_one_probability = lambda y, p: (y*np.log(p))+((1-y)*np.log(1-p))
error_for_one_probability(1,0.85) ###
error_for_one_probability(1,0.6)
error_for_one_probability(0,0.6)
error_for_one_probability(1,0.99)
error_for_one_probability(0,0.01)






######### NLP ##############
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

text = 'hi, my name is vishesh'
df_text = pd.DataFrame(iris.data, columns=iris.feature_names)
df_text['label'] = iris.target
label_names = ['hello', 'man', 'namely']
df_text['text'] = df_text['label'].apply(lambda x: 'hi my name is {}'.format(label_names[x]))
trainX, testX, trainY, testY = train_test_split(df_text, pd.get_dummies(df_text.label), test_size=.3)

TOKEN = '\\S+(?=\\s+)'
countvec = CountVectorizer(token_pattern=TOKEN, ngram_range = (1,3))
countvec_HASH = HashingVectorizer(token_pattern=TOKEN, ngram_range = (1,3), norm=None, non_negative=True)

pipeline = Pipeline(
    [
        ('countvec', countvec),
        ('logr', logr)
     ]
)


pipeline.fit(df_text['text'], df_text['label'])
pipeline.score(df_text['text'], df_text['label'])
pipeline.predict(df_text['text'])



######### NLP WITH NUMERIC/TEXT DF #########
get_text_data = FunctionTransformer(lambda x: x['text'], validate = False)
get_num_data = FunctionTransformer(lambda x: x.drop(['text', 'label'], axis = 1), validate = False)



numeric_pipeline = Pipeline([('num_only', get_num_data),
                             ('imp', Imputer())
                             ])

text_pipeline = Pipeline([('text_only', get_text_data),
                             ('cvec', CountVectorizer())])

just_numeric_data = numeric_pipeline.fit_transform(trainX)
just_text_data = text_pipeline.fit_transform(trainX)

union = FeatureUnion([('numeric_pipeline', numeric_pipeline),
                      ('text_pipeline', text_pipeline)])

# lasso = OneVsRestClassifier(Lasso(alpha=0.01, normalize=True))
# lasso.fit(trainX.drop(['text', 'label'], axis = 1), trainY)
# lasso_coef = lasso.coef_
# help(lasso_coef)
# lasso.score(iris.data, iris.target)
# lasso.predict_proba(iris.data)
OneVsRestClassifier(LogisticRegression()).get_params()
pl = Pipeline([
    ('union', union),
    ('logreg',  OneVsRestClassifier(LogisticRegression()))
    #('logreg',  OneVsRestClassifier(Lasso(alpha=0.2, normalize=True)))
])

# param_grid = {'union__text_pipeline__cvec__ngram_range':[(1,1), (1,2)]}
param_grid = {'union__text_pipeline__cvec__ngram_range':[(1,1), (1,2)],
              'logreg__estimator__C':[.1, 1, 10]}
cv = GridSearchCV(pl, param_grid=param_grid)
cv.fit(trainX, trainY)
# pl.fit(trainX, trainY)
cv.predict_proba(testX)
cv.best_params_
list(zip(pl.predict(testX), testY))
pl.score(testX, testY)


########INTERACTION############
from sklearn.preprocessing import PolynomialFeatures
df = pd.DataFrame(iris.data, columns=iris.feature_names)
pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
pf.fit_transform(df)
pf.get_feature_names()




bag = countvec.fit(df_text['text'])
bag.get_feature_names()
bag.get
bag.get
help(bag)
bag.get_params()
bag.get_stop_words()
bag.vocabulary_
bag.stop_words_

bag1 = countvec.fit_transform(df_text['text']).toarray()
countvec.get_feature_names()
print(bag1.getcol(0))
help(bag1)
bag1.astype('string')
bag1.data






#############################
#############################
#### TIME SERIES ANALYSIS ###
#############################
#############################
import pandas as pd

df1 = pd.read_csv('/Users/Vishesh/Downloads/data/AMZN.csv')
df2 = pd.read_csv('/Users/Vishesh/Downloads/data/MSFT.csv')


for df in [df1, df2]:
    df.index = pd.to_datetime(df['Date'])
    df.drop('Date', axis = 1, inplace=True)

df3 = df1.join(df2, lsuffix='_AMZN', rsuffix='_MSFT')
df3.plot()


df1['Adj Close'].resample(rule='M').sum() ###WHEN INDEX IS DATETIME

df4 = df3.join(df3.pct_change(), lsuffix='_Actual', rsuffix='_Delta')
df4.dropna(inplace=True)
df4['Adj Close_AMZN_Delta'].corr(df4['Adj Close_MSFT_Delta'])


###REGRESSION
# from scipy.stats import linregress
from statsmodels.api import OLS, add_constant
df5 = add_constant(df4)
help(OLS)
model = OLS(df5[ 'Adj Close_MSFT_Actual'], df5[['const','Adj Close_AMZN_Actual', 'Adj Close_MSFT_Delta']]).fit()
model.summary()
results = model.predict(df5[['const','Adj Close_AMZN_Actual', 'Adj Close_MSFT_Delta']])
compare_df = pd.DataFrame(list(zip(results, df5[ 'Adj Close_MSFT_Actual'])), columns = ['prediction', 'actual'])
compare_df.plot()
help(pd.tseries)




#### AUTOCORRELATION
from statsmodels.graphics.tsaplots import plot_acf


plot_acf(df5['Adj Close_MSFT_Actual'].resample(rule='Q').last(), lags = 20, alpha = .05)
help(df5['Adj Close_AMZN_Actual'].resample(rule='M').mean().autocorr())
df5['Adj Close_AMZN_Actual'].resample(rule='A').mean().pct_change().autocorr()
df5['Adj Close_AMZN_Actual'].shift(1)

df5.diff()
help(pd.DataFrame().pct_change)


####### RANDOM WALK
import statsmodels.api as sm

### RANDOM WALK
x = sm.add_constant(df5['Adj Close_AMZN_Actual'].shift(1).dropna())#.join( df5['Adj Close_AMZN_Actual'].diff().dropna(), lsuffix='_Actual', rsuffix='_Delta')
y = df5['Adj Close_AMZN_Actual'].iloc[1:]
reg = sm.OLS(y, x).fit()
reg.summary() ### RANDOM WALK SINCE SLOPE COEFF IS ALMOST 1
################# IT IS ALSO A RANDOM WALK WHEN pct_change() SLOPE IS NEAR 0

x = sm.add_constant(df5['Adj Close_AMZN_Actual'].pct_change(1).dropna())
y = df5['Adj Close_AMZN_Actual'].iloc[1:]
reg = sm.OLS(y, x).fit()
reg.summary()

###### AUGMENTED DICKEY FULLER TEST
from statsmodels.tsa.stattools import adfuller

adf_results = adfuller(df5['Adj Close_AMZN_Actual'], maxlag=10)#.resample('A').mean(), maxlag=1)
p_value = adf_results[1]
p_value
df5['Adj Close_AMZN_Actual'].resample('Q').mean().plot()
df5['Adj Close_AMZN_Actual'].diff().plot() ####CONVERTS NON STATIONARY DATA INTO STATIONARY DATA


####SIMULATING RANDOM WALK
steps = np.random.normal(loc=0, scale=10, size=500)
steps[0]=100
df_for_test = pd.DataFrame(100+np.cumsum(steps), columns=['price'])
df_for_test.plot()
df_for_test.diff().plot()####### STATIONARY WHITE NOISE
df_for_test.pct_change().plot()####### STATIONARY WHITE NOISE
adfuller(df_for_test.price)




###### AUTO CORRELATION

##AR(1) AUTO REGRESSIVE
from statsmodels.tsa.arima_process import ArmaProcess

# Plot 1: AR parameter = +0.9
'''Let ar1 represent an array of the AR parameters [1, −ϕ] as explained above. 
For now, the MA parameter array, ma1, will contain just the lag-zero coefficient of one.'''
plt.subplot(2,1,1)
ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)

# Plot 2: AR parameter = -0.9
plt.subplot(2,1,2)
ar2 = np.array([1, 0.9])
ma2 = np.array([1])
AR_object2 = ArmaProcess(ar2, ma2)
simulated_data_2 = AR_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)
plt.show()

##MA(1) MOVING AVERAGE
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf
# Plot 1: MA parameter = +0.9
'''Let ar1 represent an array of the AR parameters [1] as explained above. 
For now, the MA parameter array, ma1, will contain the lag-zero coefficient of one and theta of 0.9.'''
plt.subplot(2,1,1)
ar1 = np.array([1, ])
ma1 = np.array([1, 0.9])
MA_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = MA_object1.generate_sample(nsample=100)
# plt.plot(simulated_data_1)
plot_acf(simulated_data_1)

# Plot 2: MA parameter = -0.9
plt.subplot(2,1,2)
ar2 = np.array([1])
ma2 = np.array([1, -0.9])
MA_object2 = ArmaProcess(ar2, ma2)
simulated_data_2 = MA_object2.generate_sample(nsample=100)
plt.plot(simulated_data_2)
plt.show()

plot_acf(simulated_data_1)
plot_acf(simulated_data_2)



### TO ESTIMATE PARAMETERS FROM DATA
from statsmodels.tsa.arima_model import ARMA
mode = ARMA(simulated_data_1, order = (0,1)) #order = (2,0) means AR(2) model ## order = (0,1) means MA(1) model
mode_result = mode.fit()
mode_result.summary()
mode_result.params
mode_result.plot_predict(start = 80, end = 120)



df1mode = ARMA(df1['Adj Close'].resample('M').last().dropna(), order = (0,1))
df1mode_result = df1mode.fit()
df1mode_result.params
df1mode_result.plot_predict(start='1997-09-30', end = '2018-01-31', alpha=.05) ####FORECAST FUTURE VALUE WITH CONFIDENCE INTERVAL




#### PACF ####
from statsmodels.graphics.tsaplots import plot_pacf

df1['Adj Close'].resample('A').last().plot()
plot_pacf(df1['Adj Close'].resample('A').last(), lags = 20, alpha = 0.05) ### alpha sets width of confidence interval





##### BIC ##### LOWER BIC = BETTER MODEL
from statsmodels.tsa.arima_model import ARMA

# Fit the data to an AR(p) for p = 0,...,6 , and save the BIC
BIC = np.zeros(7)
for p in range(7):
    mod = ARMA(simulated_data_2, order=(p, 0))
    res = mod.fit()
    # Save BIC for AR(p)
    BIC[p] = res.bic

# Plot the BIC as a function of p
plt.plot(range(1, 7), BIC[1:7], marker='o')
plt.xlabel('Order of AR Model')
plt.ylabel('Baysian Information Criterion')
plt.show()



###### COINTEGRATION
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.api import OLS, add_constant

P = df5['Adj Close_AMZN_Actual']
Q = df5['Adj Close_MSFT_Actual']
coint(P,Q)

ols = OLS(Q, add_constant(P)).fit()
p_val = ols.params[1]
ad_test = adfuller(Q-p_val*P)














######### CLIMATE DATA CASE STUDY
# Import the adfuller function from the statsmodels module
from statsmodels.tsa.stattools import adfuller
temp_NY = pd.read_csv('https://assets.datacamp.com/production/course_4267/datasets/NOAA_TAVG.csv', index_col = 0)
# Convert the index to a datetime object
temp_NY.index = pd.to_datetime(temp_NY.index, format='%Y')

# Plot average temperatures
temp_NY.plot()
plt.show()

# Compute and print ADF p-value
result = adfuller(temp_NY['TAVG'])
print("The p-value for the ADF test is ", result[1])


# Import the modules for plotting the sample ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Take first difference of the temperature Series
chg_temp = temp_NY.diff()
chg_temp = chg_temp.dropna()

# Plot the ACF and PACF on the same page
fig, axes = plt.subplots(2,1)

# Plot the ACF
plot_acf(chg_temp, lags=20, ax=axes[0])

# Plot the PACF
plot_pacf(chg_temp, lags=20, ax=axes[1])
plt.show()

# Import the module for estimating an ARMA model
from statsmodels.tsa.arima_model import ARMA

# Fit the data to an AR(1) model and print AIC:
mod = ARMA(chg_temp, order=(1,0))
res = mod.fit()
print("The AIC for an AR(1) is: ", res.aic)

# Fit the data to an AR(2) model and print AIC:
mod = ARMA(chg_temp, order=(2,0))
res = mod.fit()
print("The AIC for an AR(2) is: ", res.aic)

# Fit the data to an MA(1) model and print AIC:
mod = ARMA(chg_temp, order=(0,1))
res = mod.fit()
print("The AIC for an MA(1) is: ", res.aic)

# Fit the data to an ARMA(1,1) model and print AIC:
mod = ARMA(chg_temp, order=(1,1))
res = mod.fit()
print("The AIC for an ARMA(1,1) is: ", res.aic)

res.plot_predict(start='1992-01-01', end = '2020-01-01')


# Import the ARIMA module from statsmodels
from statsmodels.tsa.arima_model import ARIMA

# Forecast interest rates using an AR(1) model
mod = ARIMA(temp_NY, order=(1,1,1))
res = mod.fit()
# Plot the original series and the forecasted series
res.plot_predict(start='1872-01-01', end='2046-01-01')
plt.show()






##############################
##############################
##########   SVM   ###########
##############################
##############################
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
# import matplotlib.pyplot as plt
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

svc = LinearSVC()
svc.fit(X_train, y_train)
svc.score(X_test, y_test)
result = svc._predict_proba_lr(X_test)
result1 = svc.decision_function(X_test)
result_compare=pd.DataFrame(result).join(pd.DataFrame(y_test, columns=['actual']))
result_decision_compare=pd.DataFrame(result1).join(pd.DataFrame(y_test, columns=['actual']))

test_1 = X_test[0]
###CALCULATE RAW MODEL OUTPUT
svc.coef_@test_1+svc.intercept_
svc._predict_proba_lr(X_test)[0]
y_test[0]



svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm.score(X_test, y_test)
# result = svc._predict_proba_lr(X_test)
result1 = svm.decision_function(X_test)
# result_compare=pd.DataFrame(result).join(pd.DataFrame(y_test, columns=['actual']))
result_decision_compare=pd.DataFrame(result1).join(pd.DataFrame(y_test, columns=['actual']))
svm.support_
svm.support_vectors_
svm.n_support_

(svm.intercept_)
(svm.dual_coef_)
svm.score(X_test, y_test)

svm_small = SVC(gamma=.5)
svm_small.fit(X_train[svm.support_], y_train[svm.support_])
svm_small.score(X_test, y_test)




##### SVC KERNELS AND PARAMS
from sklearn.model_selection import GridSearchCV
# Instantiate an RBF SVM
svm = SVC() #default kernel 'rbf'

# Instantiate the GridSearchCV object and run the search
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1], 'probability': [True, False]}
searcher = GridSearchCV(svm, param_grid=parameters)
searcher.fit(X_train,y_train)
# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))

searcher.predict_proba(X_test) ###ONLY WHEN probability = True









##############################
##############################
####### SDGClassifier ########
##############################
##############################
'''USEFUL FOR VERY LARGE DATASETS'''
from sklearn.linear_model import SGDClassifier

# We set random_state=0 for reproducibility
linear_classifier = SGDClassifier(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
             'loss':['hinge', 'log'], 'penalty':['l1', 'l2']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
