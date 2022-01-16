import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
from matplotlib.widgets import Button, Slider
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from alpha_vantage.timeseries import TimeSeries
import math
from sklearn.metrics import mean_squared_error
import datetime
from datetime import date
from datetime import date, timedelta
from datetime import datetime as date

api_key="B643EZPAZU9YAH2S" #Key used to take data from alpha_vantage library
tms = TimeSeries(key=api_key, output_format='pandas')
df, meta_data = tms.get_daily(symbol='TTM', outputsize='full')
df=df.loc[: '2019-01-01'] #Data used only for prediction from 2019 to current available date

df = df.sort_index()
df=df.reset_index()
df1=df[['date','4. close']].copy()

scaler=MinMaxScaler(feature_range=(0,1))
df2= scaler.fit_transform(np.array(df1['4. close']).reshape(-1,1))

train_size = int(len(df2)*0.70)
test_size = len(df2)-train_size

train_data, test_data = df2[0:train_size,:], df2[train_size:len(df2),:1]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a= dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)

time_step=100
X_train, y_train =  create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train= X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
X_test= X_test.reshape(X_test.shape[0],X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=50, batch_size=64, verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


look_back=100
trainPredictPlot = np.empty_like(df2)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(df2)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df2)-1, :] = test_predict



a= len(test_data) - 100

x_input=test_data[a:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output = []
n_steps = 100
i = 0
while (i < 31):

    if (len(temp_input) > 100):

        x_input = np.array(temp_input[1:])

        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))

        yhat = model.predict(x_input, verbose=0)


        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]

        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1

last_date = df1['date'].iloc[-1]
last_date = pd.to_datetime(last_date)

last_date_1 = (pd.to_datetime(last_date) + timedelta(days=1)).isoformat()

last_date_name = pd.to_datetime(last_date_1).strftime("%A")

df_date = pd.DataFrame()
for day in range(2, 47):
    if last_date_name != "Sunday" and last_date_name != "Saturday":
        df_date = df_date.append({'date': last_date_1}, ignore_index=True)

    last_date_1 = (pd.to_datetime(last_date) + timedelta(days=day)).isoformat()

    last_date_name = pd.to_datetime(last_date_1).strftime("%A")

df_date['date'] = pd.to_datetime(df_date["date"]).dt.strftime("%Y-%m-%d")


df2= scaler.inverse_transform(df2)
output = scaler.inverse_transform(lst_output)
df_date['pred'] = output

df_date['date'] = pd.to_datetime(df_date["date"]).dt.strftime("%Y-%m-%d")

df1['date'] = pd.to_datetime(df1["date"]).dt.strftime("%Y-%m-%d")
plt.plot(df1['date'], df2)
plt.plot(df1['date'],testPredictPlot)
plt.plot(df_date['date'], df_date['pred'])
plt.show()
