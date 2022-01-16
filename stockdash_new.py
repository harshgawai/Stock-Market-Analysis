import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objs as go
from plotly import tools
import matplotlib.pyplot as plt
import datetime
from datetime import date
import csv
import numpy as np
import pandas as pd
import datetime
import sklearn
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from matplotlib import style
import tweepy
from textblob import TextBlob
import nltk
from alpha_vantage.timeseries import TimeSeries
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
import base64
from datetime import date, timedelta
from datetime import datetime as date

# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
consumer_key = 'your key'
consumer_secret = 'your key'
access_token = 'your key'
access_token_secret = 'your key'

api_key = "B643EZPAZU9YAH2S"

comp = pd.read_csv('listing_status.csv')
comp = comp[comp['assetType'] == 'Stock']
comp = comp[['symbol', 'name', 'exchange']]
comp = comp.dropna(axis=0, how='any').reset_index()

dict1 = {}
for i, j in zip(comp['name'], comp['symbol']):
    dict1[i] = j

options = [{'label': i, 'value': j} for i, j in zip(dict1.keys(), dict1.values())]

app.layout = dbc.Container([
    html.H1(children='Stock Market Analysis and Prediction', style={'text-align': 'center'}),
    html.Div(children='Group -Pyffindor', style={'text-align': 'center'}),
    html.Hr(),
    dbc.FormGroup([
        html.Div([
            dbc.Label("Choose Company"),
            dcc.Dropdown(id="companies", search_value='Bank Of Nova Scotia')
        ], style={'display': 'inline-block', 'width': '50%'}), ]),

    html.Hr(),

    dbc.Row(dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H4("Stock Exchange:", className="card"),
                    html.P(id="exch", className="card-text"),

                ]
            ),

        ],
        style={"width": "20rem"},
    ), style={'padding-left': '15px', 'display': 'inline-block'}),

    html.Div([
        dbc.Label("Select Date:"),
        html.Br(),
        dcc.DatePickerRange(id='dtp', display_format='YYYY-MM-DD', start_date_placeholder_text="Start Period",
                            end_date_placeholder_text="End Period",
                            min_date_allowed=date(1999, 11, 1), initial_visible_month=date(2020, 12, 1))
    ], style={'display': 'inline-block', 'margin-left': '40%'}),

    html.Hr(),
    dbc.Button('Visualize', id='example-button', color='primary', style={'margin-bottom': '1em', 'margin-left': '46%'}),
    dcc.Graph(id='mygraph', animate=True, style={'background-color': '#1a2d46', 'color': '#ffffff'}),
    dcc.Graph(id='mygraph2', style={'background-color': '#1a2d46', 'color': '#ffffff'}),

    html.Hr(),

    dbc.FormGroup([
        html.Div([html.H4("Maximum 'open' price of the stocks:", className='mo'),
                  html.P(id='maxo', className='c-mo'), ],
                 style={'width': '50%', 'display': 'inline-block'}),

        html.Div([html.H4("Maximum 'close' price of the stocks:", className='mc'),
                  html.P(id='maxc', className='c-mc')],
                 style={'width': '50%', 'display': 'inline-block'})
    ]),
    html.Hr(),
    html.H3('Word Cloud:'),

    html.Img(id="image_wc", style={'margin-left': '30%'}),
    html.Hr(),

    dcc.Markdown(id='para', style={"white-space": "pre", "overflow-x": "scroll", "overflow-y": "scroll"}),

    html.Hr(),
    dbc.FormGroup([
        html.H4("Choose Machine Learning Model:", style={'padding-left': '120px'}),
        dcc.Dropdown(id="vis2", value=1,
                     options=[{"label": "DecisionTreeRegressor", "value": 1},
                              {"label": "LinearRegression", "value": 2},
                              {"label": "RandomForestRegressor", "value": 3},
                              ],
                     style={'margin-left': '20px'}),
    ], style={'margin-left': 'auto', 'margin-right': 'auto', 'width': '50%'}),
    dbc.Button('Predict', id='example-button2', color='primary', style={'margin-bottom': '1em', 'margin-left': '50%'}),
    dcc.Graph(id='predict'),

])


@app.callback(
    [Output('companies', 'options'),
     Output('exch', 'children')],
    [Input('companies', 'search_value'),
     Input('companies', 'value')]

)
def update_options(search_value, val):
    exc = comp[comp['symbol'] == val]['exchange']

    return [i for i in options if search_value in i['label']], exc


@app.callback(
    [Output('mygraph', 'figure'), Output('mygraph2', 'figure'), Output('image_wc', 'src'),
     Output('maxo', 'children'), Output('maxc', 'children'), Output('para', 'children')],
    [Input('example-button', 'n_clicks')],
    [State('companies', 'value'),
     State('dtp', 'start_date'), State('dtp', 'end_date')]
)
def show_vis(n_clicks, companies, s_date, e_date):
    if companies == None:
        raise PreventUpdate
    comp_name = comp[comp['symbol'] == companies]['name']

    tms = TimeSeries(key=api_key, output_format='pandas')
    df, meta_data = tms.get_daily(symbol=companies, outputsize='full')
    df = df.reset_index()

    mask = df[(df['date'] > s_date) & (df['date'] <= e_date)]

    max_o = np.max(mask['1. open'])
    max_c = np.max(mask['4. close'])

    fig = go.Figure(go.Candlestick(x=mask['date'], open=mask['1. open'],
                                   high=mask['2. high'], low=mask['3. low'],
                                   close=mask['4. close']))

    window_size = len(mask) // 10
    bac = mask['4. close'].rolling(window=window_size).mean()

    fig2 = go.Figure(go.Scatter(x=mask['date'], y=bac, name='BAC'))
    fig2.add_trace(go.Scatter(x=mask['date'], y=mask['4. close'], name='Close'))

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    tweet_search = str(list(comp_name)[0])
    # tweet_search1=tweet_search+str(' '+'stock price')

    public_tweets = api.search(tweet_search)
    txt = []
    sentimental_values = []

    for tweet in public_tweets:
        analysis = TextBlob(tweet.text)
        txt.append(str(analysis))
        sentimental_values.append(str(analysis.sentiment.polarity))
    text = ''
    for i in txt:
        text = text + i

    word_cloud = WordCloud(background_color='white', max_words=20, stopwords=set(STOPWORDS))
    mycloud = word_cloud.generate(text)
    img_wc = mycloud.to_image()
    img = BytesIO()
    img_wc.save(img, format='PNG')

    return fig, fig2, 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode()), max_o, max_c, txt,


@app.callback(
    [Output('predict', 'figure')],
    Input('example-button2', 'n_clicks'),
    [State('companies', 'value'), State('vis2', 'value')]
)
def pred(n_clicks, companies, val):
    if companies == None:
        raise PreventUpdate
    tms = TimeSeries(key=api_key, output_format='pandas')
    df, meta_data = tms.get_daily(symbol=companies, outputsize='full')
    df = df.loc[: '2019-01-01']
    df = df.sort_index()
    df = df.reset_index()
    df2 = df[['4. close', 'date']].copy()

    future_days = 30
    df2['Prediction'] = df2[['4. close']].shift(-future_days)

    X = np.array(df2.drop(['Prediction', 'date'], 1))[:-future_days]
    y = np.array(df2['Prediction'])[:-future_days]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    if val == 1:
        model = DecisionTreeRegressor().fit(X_train, y_train)
    if val == 2:
        model = LinearRegression().fit(X_train, y_train)
    if val == 3:
        model = RandomForestRegressor().fit(X_train, y_train)

    print(df2)
    print("=========")

    last_date = df2['date'].iloc[-1]
    last_date = pd.to_datetime(last_date)
    last_date_1 = (pd.to_datetime(last_date) + timedelta(days=1)).isoformat()
    last_date_name = pd.to_datetime(last_date_1).strftime("%A")

    for day in range(2, 47):
        if last_date_name != "Sunday" and last_date_name != "Saturday":
            df2 = df2.append({'date': last_date_1}, ignore_index=True)
            close_price = df2['4. close'].iloc[-day]
            df2 = df2.append({'4. close': close_price}, ignore_index=True)
        last_date_1 = (pd.to_datetime(last_date) + timedelta(days=day)).isoformat()
        last_date_name = pd.to_datetime(last_date_1).strftime("%A")

    future_days = 94
    future_pred = df2.fillna(0)
    future_pred = future_pred.drop(['Prediction', 'date'], 1)[:-future_days]
    future_pred = future_pred.tail(future_days)
    future_pred = np.array(future_pred)

    valid = df[X.shape[0]:]

    last_date = valid['date'].iloc[-1]
    last_date = pd.to_datetime(last_date)
    last_date_1 = (pd.to_datetime(last_date) + timedelta(days=1)).isoformat()
    last_date_name = pd.to_datetime(last_date_1).strftime("%A")

    for day in range(2, 47):
        if last_date_name != "Sunday" and last_date_name != "Saturday":
            valid = valid.append({'date': last_date_1}, ignore_index=True)
            valid = valid.append({'date': last_date_1}, ignore_index=True)
        last_date_1 = (pd.to_datetime(last_date) + timedelta(days=day)).isoformat()
        last_date_name = pd.to_datetime(last_date_1).strftime("%A")

    valid['Predictions'] = model.predict(future_pred)
    """
    predictions = model_prediction.copy()

    print(len(predictions))
    print(len(valid['Predictions']))

    valid['Predictions'] = predictions

    print(valid)
    print("=========")
    print(valid.columns)

    #====================================================================================================================

    # creating dataframe
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

    # setting index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

    # creating train and test sets
    dataset = new_data.values

    train = dataset[0:987, :]
    valid = dataset[987:, :]

    # converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    # predicting 246 values, using past 60 from the train data
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    train = new_data[:987]
    valid = new_data[987:]
    valid['Predictions'] = closing_price
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    #====================================================================================================================
    """
    print(valid)
    print("=========")
    print(valid.columns)

    fig = go.Figure(go.Scatter(x=df['date'], y=df2['4. close'], name='Original Training Price'))
    fig.add_trace(go.Scatter(x=valid['date'], y=valid['4. close'], name='Original Test Price'))
    fig.add_trace(go.Scatter(x=valid['date'], y=valid['Predictions'], name='Predicted Test Price'))

    return fig,


app.run_server(debug=True, port=8051)
