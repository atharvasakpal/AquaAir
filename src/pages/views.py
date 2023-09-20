from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import requests
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create your views here.
def home_view(request,*args, **kwargs):
    return render(request, "Homepage.html",{})
def AQI(request,*args, **kwargs):
    return render(request, "AQI.html",{})

def WQI(request,*args, **kwargs):
    return render(request, "WQI.html", {})

def AQI_view(request,*args,**kwargs):
    location = request.GET['Location']
    apikey = "5c851b011e0f071f8e0496731dd999e3";
    ###taking input as city name, for now typed manually

    city = str(location)

    ##lat lon api code
    response1 = requests.get(
        'http://api.openweathermap.org/geo/1.0/direct?q={}&appid=0995af22d236b1cc2e36ef198a08fb29'.format(city))
    data1 = response1.json()
    ###got values of its latitude and longitude and stored it in some variable
    lati = data1[0]['lat']
    long = data1[0]['lon']
    ###latit = {'lat' : lati}
    ###longit = {'lon' : long}
    ###passing lat and log to the OpenWeather api
    url = 'http://api.openweathermap.org/data/2.5/air_pollution/history?lat={}&lon={}&start=1546281000&end=1693506600&appid=0995af22d236b1cc2e36ef198a08fb29'.format(
        lati, long)
    response = requests.get(url)

    response2 = requests.get(
        'https://api.airvisual.com/v2/nearest_city?lat={}&lon={}&key=3a7d2c85-6a18-4fd6-8c5c-9a00df1004d5'.format(lati,
                                                                                                                  long))
    mf = pd.DataFrame(response2.json()['data'])
    aqi = mf.current.pollution['aqius']
    aqi = str(aqi)

    ##coverting json to a pandas dataframe
    df = pd.DataFrame(response.json()['list'])
    dataAQI = pd.DataFrame.from_records(df['main'])
    data = pd.DataFrame.from_records(df['components'])
    data['date'] = df.loc[:, 'dt']
    data['AQI'] = dataAQI.loc[:, 'aqi']
    data['date'] = pd.to_datetime(data['date'], unit='s')
    grouped = data.groupby(pd.Grouper(key='date', freq='D'))
    daily_data = grouped.mean()
    daily_data = daily_data.dropna()
    daily_data = daily_data[['pm2_5', 'pm10', 'no2', 'so2']]

    healthy_pm2_5 = 25
    healthy_pm10 = 50
    healthy_no2 = 25
    healthy_so2 = 20

    figBar = make_subplots(
        rows=2, cols=2,
        subplot_titles=("pm2.5", "pm10", "no2", 'so2'))

    figBar.add_trace(go.Bar(y=[healthy_pm2_5, daily_data.pm2_5.mean()], x=['Healthy', 'Measured']),
                     row=1, col=1)

    figBar.add_trace(go.Bar(y=[healthy_pm10, daily_data.pm10.mean()], x=['Healthy', 'Measured']),
                     row=1, col=2)

    figBar.add_trace(go.Bar(y=[healthy_no2, daily_data.no2.mean()], x=['Healthy', 'Measured']),
                     row=2, col=1)

    figBar.add_trace(go.Bar(y=[healthy_so2, daily_data.so2.mean()], x=['Healthy', 'Measured']),
                     row=2, col=2)

    figBar.update_layout(height=500, width=700,
                         title_text="Healthy range vs Measured Values")

    #figBar.show()
    figBarhtml = figBar.to_html()
    # buffer = BytesIO()
    # plt.savefig(buffer,format='png')
    # buffer.seek(0)
    # image_png= buffer.getvalue()
    # graph= base64.b64encode(image_png)
    # graph= graph.decode('utf-8')
    # buffer.close()



    labels = ['pm2.5', 'pm10', 'no2', 'so2']
    valuesH = [healthy_pm2_5, healthy_pm10, healthy_pm2_5, healthy_no2, healthy_so2]
    valuesM = [daily_data.pm2_5.mean(), daily_data.pm10.mean(), daily_data.no2.mean(), daily_data.so2.mean()]

    figPie = make_subplots(
        rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=("Healthy", "Measured"))

    figPie.add_trace(
        go.Pie(values=[healthy_pm2_5, healthy_pm10, healthy_pm2_5, healthy_no2, healthy_so2], labels=labels,
               domain=dict(x=[0, 0.5])),
        row=1, col=1)

    figPie.add_trace(
        go.Pie(values=[daily_data.pm2_5.mean(), daily_data.pm10.mean(), daily_data.no2.mean(), daily_data.so2.mean()],
               labels=labels, domain=dict(x=[0.5, 1])),
        row=1, col=2)

    figPie.update_layout(title_text="Healthy vs Measured Composition")
    figPiehtml = figPie.to_html()
    #figPie.show()

    fig = make_subplots(rows=2, cols=2, subplot_titles=('pm2.5', 'pm10', 'so2', 'no2'))
    fig.append_trace(go.Scatter(x=daily_data.index, y=daily_data['pm2_5']), row=1, col=1)
    fig.append_trace(go.Scatter(x=daily_data.index, y=daily_data['pm10']), row=1, col=2)
    fig.append_trace(go.Scatter(x=daily_data.index, y=daily_data['so2']), row=2, col=1)
    fig.append_trace(go.Scatter(x=daily_data.index, y=daily_data['no2']), row=2, col=2)
    fig1 = fig.to_html()
    #fig.show()

    df_input = daily_data[['pm2_5', 'pm10', 'so2', 'no2']]

    df_input['Month'] = [i.month for i in df_input.index]
    df_input['Year'] = [i.year for i in df_input.index]
    df_input['DayOfYear'] = [i.dayofyear for i in df_input.index]

    features = df_input[['Month', 'Year', 'DayOfYear']]
    target = df_input[['pm2_5', 'pm10', 'so2', 'no2']]
    x_train, x_test, y_train, y_test = train_test_split(features, target)
    reg = xgb.XGBRegressor(n_estimators=100)
    reg.fit(x_train, y_train)

    predictions = pd.DataFrame(reg.predict(x_test), columns=['Ppm2_5', 'Ppm10', 'Pso2', 'Pno2'])
    predictions = predictions.set_index(x_test.index)

    futureDate = pd.DataFrame(pd.date_range(start='2023-09-01', end='2025-01-01'), columns=['date'])
    futureDate = futureDate.set_index(['date'])
    futureDate['Month'] = [i.month for i in futureDate.index]
    futureDate['Year'] = [i.year for i in futureDate.index]
    futureDate['DayOfYear'] = [i.dayofyear for i in futureDate.index]

    results = pd.DataFrame(reg.predict(futureDate), columns=['pm2_5', 'pm10', 'so2', 'no2'])
    results = results.set_index(futureDate.index)

    final = pd.concat([df_input, results])

    fig = make_subplots(rows=2, cols=2, subplot_titles=('pm2.5', 'pm10', 'so2', 'no2'))
    fig.append_trace(go.Scatter(x=final.index, y=final['pm2_5']), row=1, col=1)
    fig.append_trace(go.Scatter(x=final.index, y=final['pm10']), row=1, col=2)
    fig.append_trace(go.Scatter(x=final.index, y=final['so2']), row=2, col=1)
    fig.append_trace(go.Scatter(x=final.index, y=final['no2']), row=2, col=2)
    fig.add_vline(x='2023-08-30', line_dash='dash', line_color='black')
    fig.add_vrect(x0='2023-08-30', x1='2025-01-01', fillcolor='green', opacity=0.2, annotation_text='predicts')
    fightml= fig.to_html()
    #fig.show()

    context= {"figBar": figBarhtml, "fig1": fig1, "figPie": figPiehtml,"fig":fightml, "location": location,"AQI": aqi}
    return render(request, 'AQIresult.html', context)

def WQI_view(request,*args,**kwargs):

    data = pd.read_csv('water_potability.csv')
    df = pd.DataFrame(data)

    data['ph'].fillna(data['ph'].mean(), inplace=True)
    data['Hardness'].fillna(data['Hardness'].mean(), inplace=True)
    data['Solids'].fillna(data['Solids'].mean(), inplace=True)
    data['Sulfate'].fillna(data['Sulfate'].mean(), inplace=True)
    data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean(), inplace=True)
    data['Chloramines'].fillna(data['Chloramines'].mean(), inplace=True)
    data['Conductivity'].fillna(data['Conductivity'].mean(), inplace=True)
    data['Organic_carbon'].fillna(data['Organic_carbon'].mean(), inplace=True)
    data['Turbidity'].fillna(data['Turbidity'].mean(), inplace=True)

    X = df.drop(columns=['Potability'])
    y = df['Potability']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)

    y_pred_rf = random_forest.predict(X_test)



    Sulfate = request.GET['Sulphur']
    ph = request.GET['pH']
    Solids = request.GET['solids']
    Hardness = request.GET['hardness']
    Trihalomethanes = request.GET['trihalomethanes']
    Chloramines = request.GET["Chloramines"]
    Conductivity = request.GET["conductivity"]
    Organic_carbon = request.GET["organiccarbon"]
    Turbidity = request.GET["turbidity"]

    input_features = np.array(
        [[Sulfate, ph, Solids, Hardness, Trihalomethanes, Chloramines, Conductivity, Organic_carbon, Turbidity]])

    prediction = random_forest.predict(input_features)
    if (prediction[0] == 0):
        str = "<h1>Water is not potable</h1>"
    else:
        str = "<h1>Water is potable</h1>"

    return render(request, 'WQIresult.html', {'str': str})





