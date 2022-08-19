import pandas as pd
import numpy as np
import chart_studio.plotly as py
import cufflinks as cf
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
import plotly.graph_objects as go

# pip install plotly
# For Excel file use
# info = pd.read_excel('titles.xlsx')
# Convert info into DataFrame using
# df = pd.DataFrame(info)


# PIE CHART In-Build DataFrame in plotly.express .data.stocks

df = px.data.stocks()
fig = px.pie(df, values='date', names='AAPL')
fig.show()

# LINE CHART and Graph_Objects
# import plotly.graph_objects as go
# Single line graph

fig = px.line(df, x='date', y='GOOG', labels={'x': 'Date', 'y': 'Price'})

# Multiple line graphs

fig = px.line(df, x='date', y=['GOOG', 'AAPL'], labels={'x': 'Date', 'y': 'Price'}, title='Apple Vs. Google')

# Adding traces to line plot with graph_objects (Can  Have different modes for traces like ['lines','lines+markers'])
# go.Figure() is used for customization

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.date, y=df.AAPL,
                         mode='lines', name='Apple'))
fig.add_trace(go.Scatter(x=df.date, y=df.AMZN,
                         mode='lines+markers', name='Amazon'))

# You can create custom lines (Dashes : dash, dot, dashdot)

fig.add_trace(go.Scatter(x=df.date, y=df.GOOG,
                         mode='lines+markers', name='Google',
                         line=dict(color='firebrick', width=2, dash='dashdot')))
fig.show()

# Updating layout of the line simple

fig.update_layout(title='Stock Price Data 2018 - 2020',
                  xaxis_title='Price',
                  yaxis_title='Date')
fig.show()

# Complex Update_Layout

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False, ),
    autosize=False,
    margin=dict(
        autoexpand=False, l=100, r=20, t=110,
    ),
    showlegend=False,
    plot_bgcolor='white',
)

fig.show()

# BAR CHART and Can give Custom Query

df_us = px.data.gapminder().query("country == 'United States'")
fig = px.bar(df_us, x='year', y='pop')
fig.show()

# Stacked BAR CHART on any specific field

df_tips = px.data.tips()
fig = px.bar(df_tips, x='day', y='tip', color='sex', title='Tips by Sex on Each Day',
             labels={'tip': 'Tip Amount', 'day': 'Day of the Week'})
fig.show()

# Using different BARMODE (group=places next to each other)

fig = px.bar(df_tips, x='day', y='tip', color='sex', title='Tips by Sex on Each Day',
             labels={'tip': 'Tip Amount', 'day': 'Day of the Week'}, barmode='group')

# Putting text on graph

df_europe = px.data.gapminder().query("continent == 'Europe' and year == 2007 and pop > 2.e6")
fig = px.bar(df_europe, y='pop', x='country', text='pop', color='country')

# Putting the text outside the bar text size to 2 values

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

# Setting Font Size

fig.update_layout(uniformtext_minsize=8)

# Rotate label by 45 degree

fig.update_layout(xaxis_tickangle=-45)

# SCATTER PLOT with Hover Data

df_iris = px.data.iris()
fig = px.scatter(df_iris, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])
fig.show()

# Black marker edges with width 2 , also show the scale on the right "go.Figure() used for complex manipulation"

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_iris.sepal_width, y=df_iris.sepal_length,
    mode='markers',
    marker_color=df_iris.sepal_width,
    text=df_iris.species,
    marker=dict(showscale=True)
))
fig.update_traces(marker_line_width=2, marker_size=10)

# For lots of data use Scattergl complex scatter plot for a scatter fig

fig = go.Figure(data=go.Scattergl(
    x=np.random.randn(100000),
    y=np.random.randn(100000),
    mode='markers',
    marker=dict(
        color=np.random.randn(100000),
        colorscale='Viridis',
        line_width=1
    )
))

fig.show()

# PIE CHART complex "Color scheme = plotly.com/python/buildin-colorscales/"

df_pop = px.data.gapminder().query("year == 2007").query("continent == 'Asia'")
fig = px.pie(df_pop, values='pop', names='country',
             title='Population of Asian continent',
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

# Customize new PIE CHART with 6 colors corresponding to the labels
# Values is the total amount for each label

colors = ['blue', 'green', 'black', 'purple', 'red', 'brown']
fig = go.Figure(data=[go.Pie(labels=['Water', 'Grass', 'Normal', 'Psychic', 'Fire', 'Ground'],
                             values=[110, 90, 80, 80, 70, 60])])

# Define hover_info , text size, pull amount and stroke

fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  textinfo='label+percent', pull=[0.1, 0, 0.2, 0, 0, 0],
                  marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))
fig.show()

# HISTOGRAM

dice_1 = np.random.randint(1, 7, 5000)
dice_2 = np.random.randint(1, 7, 5000)
dice_sum = dice_1 + dice_2

# BINS-number of bars and marginal-Another violin plot on top
# Value Label is for x axis

fig = px.histogram(dice_sum, nbins=11, labels={'value': 'Dice Roll'},
                   title='5000 Dice Roll Histogram', marginal='violin',
                   color_discrete_sequence=['green'])

# Customize Histogram fig , BarGap is used to give Gap between bars

fig.update_layout(
    xaxis_title_text='Dice Roll',
    yaxis_title_text='Dice Sum',
    bargap=0.2, showlegend=False
)

fig.show()

# Stacked Histogram based on color

df_tips = px.data.tips()
fig = px.histogram(df_tips, x="total_bill", color="sex")
fig.show()

# BOX PLOT with all points

df_tips = px.data.tips()
fig = px.box(df_tips, x='sex', y='tip', points='all')
fig.show()

# Multi BOX PLOT based on (sex/any specific field)

fig = px.box(df_tips, x='day', y='tip', color='sex')
fig.show()

# Adding Standard deviation to BOX PLOT

fig = go.Figure()
fig.add_trace(go.Box(x=df_tips.sex, y=df_tips.tip, marker_color='blue',
                     boxmean='sd'))

# Complex Styling

df_stocks = px.data.stocks()
fig = go.Figure()

# Show all points, spread them so they don't overlap (jitter) and change whisker width

fig.add_trace(go.Box(y=df_stocks.GOOG, boxpoints='all', name='Google',
                     fillcolor='blue', jitter=0.5, whiskerwidth=0.2))
fig.add_trace(go.Box(y=df_stocks.AAPL, boxpoints='all', name='Apple',
                     fillcolor='red', jitter=0.5, whiskerwidth=0.2))

# Change background / grid colors

fig.update_layout(title='Google vs. Apple',
                  yaxis=dict(gridcolor='rgb(255, 255, 255)',
                             gridwidth=3),
                  paper_bgcolor='rgb(243, 243, 243)',
                  plot_bgcolor='rgb(243, 243, 243)')
fig.show()

# VIOLIN PLOT

df_tips = px.data.tips()
fig = px.violin(df_tips, y="total_bill", box=True, points='all')

# Multiple VIOLIN PLOT based on specific color/field

fig = px.violin(df_tips, y="tip", x="smoker", color="sex", box=True, points="all",
                hover_data=df_tips.columns)

# Morphing left and right side based on condition on axis

fig = go.Figure()
fig.add_trace(go.Violin(x=df_tips['day'][df_tips['smoker'] == 'Yes'],
                        y=df_tips['total_bill'][df_tips['smoker'] == 'Yes'],
                        legendgroup='Yes', scalegroup='Yes', name='Yes',
                        side='negative',
                        line_color='blue'))
fig.add_trace(go.Violin(x=df_tips['day'][df_tips['smoker'] == 'No'],
                        y=df_tips['total_bill'][df_tips['smoker'] == 'No'],
                        legendgroup='Yes', scalegroup='Yes', name='No',
                        side='positive',
                        line_color='red'))

# Density Heat Map using seaborn data

flights = sns.load_dataset("flights")
fig = px.density_heatmap(flights, x='year', y='month', z='passengers',
                         color_continuous_scale="Viridis")

# Adding HISTOGRAM

fig = px.density_heatmap(flights, x='year', y='month', z='passengers',
                         marginal_x="histogram", marginal_y="histogram")
fig.show()

# 3D SCATTER PLOT using flights data from seaborn

fig = px.scatter_3d(flights, x='year', y='month', z='passengers', color='year',
                    opacity=0.7, width=800, height=400)
fig.show()

# 3D LINE PLOT using flights data from seaborn

fig = px.line_3d(flights, x='year', y='month', z='passengers', color='year')
fig.show()

# SCATTER MATRIX used to compare changes when comparing column data

fig = px.scatter_matrix(flights, color='month')
fig.show()

# MAP SCATTER PLOT color gives classification

df = px.data.gapminder().query("year == 2007")
fig = px.scatter_geo(df, locations="iso_alpha",
                     color="continent",
                     hover_name="country",
                     size="pop",
                     projection="orthographic")
fig.show()

# POLAR CHART LINE-POLAR

df_wind = px.data.wind()
fig = px.line_polar(df_wind, r="frequency", theta="direction", color="strength",
                    line_close=True, template="plotly_dark", width=800, height=400)
fig.show()

# TERNARY PLOT 3 points of a triangle (a,b,c)

df_exp = px.data.experiment()
px.scatter_ternary(df_exp, a="experiment_1", b="experiment_2",
                   c='experiment_3', hover_name="group", color="gender")

# ANIMATED PLOT SCATTER PLOT (Animation frame based on Year , Animation group based on country)
# Custom Range for x and y axis

df_cnt = px.data.gapminder()
fig = px.scatter(df_cnt, x="gdpPercap", y="lifeExp", animation_frame="year",
                 animation_group="country",
                 size="pop", color="continent", hover_name="country",
                 log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])


# ANIMATED PLOT BAR PLOT

fig = px.bar(df_cnt, x="continent", y="pop", color="continent",
             animation_frame="year", animation_group="country", range_y=[0,4000000000])
fig.show()




