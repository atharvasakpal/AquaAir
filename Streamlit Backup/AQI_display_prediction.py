## All the code regarding AQI page is defined in functions here. called in Home.py

import pandas as pd
import numpy as np
import requests
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb

healthy_pm2_5 = 5
healthy_pm10 = 50
healthy_no2 = 25
healthy_so2 = 40

def show_aqi(city):
    response1 = requests.get('http://api.openweathermap.org/geo/1.0/direct?q={}&appid=0995af22d236b1cc2e36ef198a08fb29'.format(city))
    data1 = response1.json()
    ###got values of its latitude and longitude and stored it in some variable
    lati = data1[0]['lat']
    long = data1[0]['lon']

    response2 = requests.get('https://api.airvisual.com/v2/nearest_city?lat={}&lon={}&key=3a7d2c85-6a18-4fd6-8c5c-9a00df1004d5'.format(lati,long))
    df = pd.DataFrame(response2.json()['data'])
    aqi = df.current.pollution['aqius']
    return aqi

def load_dataset(city):
    response1 = requests.get('http://api.openweathermap.org/geo/1.0/direct?q={}&appid=0995af22d236b1cc2e36ef198a08fb29'.format(city))
    data1 = response1.json()
    ###got values of its latitude and longitude and stored it in some variable
    lati = data1[0]['lat']
    long = data1[0]['lon']
    ###latit = {'lat' : lati}
    ###longit = {'lon' : long}
    ###passing lat and log to the OpenWeather api
    url='http://api.openweathermap.org/data/2.5/air_pollution/history?lat={}&lon={}&start=1546281000&end=1694882874&appid=0995af22d236b1cc2e36ef198a08fb29'.format(lati,long)
    response = requests.get(url)

    ##coverting json to a pandas dataframe
    df = pd.DataFrame(response.json()['list'])
    dataAQI =  pd.DataFrame.from_records(df['main'])
    data = pd.DataFrame.from_records(df['components'])
    data['date'] = df.loc[:,'dt']
    data['AQI'] = dataAQI.loc[:,'aqi']
    data['date'] = pd.to_datetime(data['date'],unit='s')
    grouped = data.groupby(pd.Grouper(key='date', freq='D'))
    daily_data = grouped.mean()
    daily_data = daily_data.dropna()
    daily_data = daily_data[['pm2_5','pm10','no2','so2']]
    return daily_data

def healthyBarPlot(daily_data):
    
    
    figBar = make_subplots(
    rows=2, cols=2,
    subplot_titles=("pm2.5", "pm10", "no2", 'so2'))

    figBar.add_trace(go.Bar(y=[daily_data.pm2_5[-30:-1].mean(),daily_data.pm2_5[-1]],x=['Average','Previous Day']),
              row=1, col=1)

    figBar.add_trace(go.Bar(y=[daily_data.pm10[-30:-1].mean(),daily_data.pm10[-1]],x=['Average','Previous Day']),
              row=1, col=2)

    figBar.add_trace(go.Bar(y=[daily_data.no2[-30:-1].mean(),daily_data.no2[-1]],x=['Average','Previous Day']),
              row=2, col=1)

    figBar.add_trace(go.Bar(y=[daily_data.so2[-30:-1].mean(),daily_data.so2[-1]],x=['Average','Previous Day']),
              row=2, col=2)

    figBar.update_layout(height=500, width=700,
                  title_text="30 Day Mean vs Previous Day Value")
    return figBar


def healthyPiePlot(daily_data):
    labels=['pm2.5','pm10','no2','so2']
    
    figPie = make_subplots(
    rows=1, cols=2,specs=[[{"type": "pie"}, {"type": "pie"}]],
    subplot_titles=("Average", "Measured"))

    figPie.add_trace(go.Pie(values=[daily_data.pm2_5[-30:-1].mean(),daily_data.pm10[-30:-1].mean(),daily_data.no2[-30:-1].mean(),daily_data.so2[-30:-1].mean()],labels=labels, domain=dict(x=[0, 0.5])),
              row=1, col=1)

    figPie.add_trace(go.Pie(values=[daily_data.pm2_5[-1],daily_data.pm10[-1],daily_data.no2[-1],daily_data.so2[-1]],labels=labels, domain=dict(x=[0.5, 1])),
              row=1, col=2)

    figPie.update_layout(title_text="10 Day Mean vs Previous Day")
    return figPie

def pastPlot(daily_data):
    fig = make_subplots(rows=2,cols=2,subplot_titles=('pm2.5','pm10','so2','no2'))
    fig.append_trace(go.Scatter(x=daily_data.index,y=daily_data['pm2_5']),row=1,col=1)
    fig.append_trace(go.Scatter(x=daily_data.index,y=daily_data['pm10']),row=1,col=2)
    fig.append_trace(go.Scatter(x=daily_data.index,y=daily_data['so2']),row=2,col=1)
    fig.append_trace(go.Scatter(x=daily_data.index,y=daily_data['no2']),row=2,col=2)
    fig.update_layout(title_text="Trends every year")
    return fig

def mlModel(daily_data):
    df_input = daily_data[['pm2_5','pm10','so2','no2']]
    df_input['Month'] = [i.month for i in df_input.index]
    df_input['Year'] = [i.year for i in df_input.index]
    df_input['DayOfYear'] = [i.dayofyear for i in df_input.index]
    ##Training the model
    features = df_input[['Month','Year','DayOfYear']]
    target = df_input[['pm2_5','pm10','so2','no2']]
    x_train, x_test, y_train, y_test = train_test_split(features,target)
    reg = xgb.XGBRegressor(n_estimators=100)
    reg.fit(x_train,y_train)
    ##predictions

    predictions = pd.DataFrame(reg.predict(x_test),columns=['Ppm2_5','Ppm10','Pso2','Pno2'])
    predictions = predictions.set_index(x_test.index)


    futureDate = pd.DataFrame(pd.date_range(start='2023-09-01',end='2025-01-01'),columns=['date'])
    futureDate = futureDate.set_index(['date'])
    futureDate['Month'] = [i.month for i in futureDate.index]
    futureDate['Year'] = [i.year for i in futureDate.index]
    futureDate['DayOfYear'] = [i.dayofyear for i in futureDate.index]

    results = pd.DataFrame(reg.predict(futureDate),columns=['pm2_5','pm10','so2','no2'])
    results = results.set_index(futureDate.index)
    final = pd.concat([df_input,results])
    
    #plotting predictions
    figPred = make_subplots(rows=2,cols=2,subplot_titles=('pm2.5','pm10','so2','no2'))
    figPred.append_trace(go.Scatter(x=final.index,y=final['pm2_5']),row=1,col=1)
    figPred.append_trace(go.Scatter(x=final.index,y=final['pm10']),row=1,col=2)
    figPred.append_trace(go.Scatter(x=final.index,y=final['so2']),row=2,col=1)
    figPred.append_trace(go.Scatter(x=final.index,y=final['no2']),row=2,col=2)
    figPred.add_vline(x='2023-08-30',line_dash='dash',line_color='black')
    figPred.add_vrect(x0='2023-08-30',x1='2025-01-01',fillcolor='green',opacity=0.2,annotation_text='predicts')
    figPred.update_layout(title_text='Future Predicted Values using XGboost')
    return figPred





