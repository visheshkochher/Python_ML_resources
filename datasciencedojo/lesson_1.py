import pandas as pd
import numpy as np
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.datasets import load_iris, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
#
# iris = load_iris()
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df.describe()
# df.groupby('Species').count()
# df['Species'] = iris.target
# df['Species'] = df['Species'].map({0: 'setosa',
#                                    1: 'versicolor',
#                                    2: 'virginica'})


def plot_box(df, split, metric='count'):
    groups = df[split].unique()
    traces = []
    for group in groups:
        y = df[df[split] == group][metric]
        trace = go.Box(y=y,
                       name=group
                       )
        traces.append(trace)
    data = traces
    plotly.offline.plot(data)


def plot_scatter(df, x_series, y_series, color_col):
    groups = df[color_col].unique()
    traces = []
    for group in groups:
        x_data = df[df[color_col] == group][x_series]
        y_data = df[df[color_col] == group][y_series]
        trace = go.Scatter(x=x_data,
                           y=y_data,
                           name=group,
                           mode='markers',
                           marker=dict(
                               size='16',
                               color=group,
                               colorscale='Virdis',
                               opacity=0.8
                           )
                           )
        traces.append(trace)
    layout = go.Layout(
        title='Scatter Plot',
        hovermode='closest',
        xaxis=dict(title=x_series),
        yaxis=dict(title=y_series)
    )
    data = traces
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig)



###DENSITY PLOTS
def plot_density(df, x, series_group):
    # df[x] = df['Age'].dropna(axis = 0)
    data_list = []
    unique_groups = df[series_group].unique()
    for sub_group in unique_groups:
        data_list.append(df[df[series_group] == sub_group][x])
    fig = ff.create_distplot(data_list, unique_groups, bin_size=.05, show_hist=False)
    fig['layout']['xaxis'].update(title=x)
    # fig['layout']['yaxis'].update(title='Density')
    fig['layout'].update(title='Density Plot')
    plotly.offline.plot(fig)



### FACET PLOTS
def plot_facet_scatter(df, x, y, group1, group2):
    fig = ff.create_facet_grid(
        df,
        x=x,
        y=y,
        color_name=group2,
        color_is_cat=True if df[group2].dtype.name == 'category' else False,
        # facet_col=group1,
        facet_row=group1,
    )
    plotly.offline.plot(fig)

# plot_box(df, 'Species', 'sepal length (cm)')


# plot_scatter(df, 'sepal length (cm)', 'sepal width (cm)', 'Species')
# plot_scatter(df, 'petal length (cm)', 'sepal width (cm)', 'Species')


# plot_density(df, 'petal width (cm)', 'Species')



#
# boston = load_boston()
# df1 = pd.DataFrame(boston.data, columns=boston.feature_names)
# df1['Target'] = boston.target
# df1['RAD'] = df1['RAD'].astype('category')
# plot_box(df1, 'RAD', 'Target')

#
# ###PAIR PLOTS
# sns.pairplot(df1)
# plt.show()
#
#


# plot_facet_scatter(df1, 'AGE', 'Target', 'RAD', 'ZN')