import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import requests
from sklearn.model_selection import train_test_split
#Scaling data to ensure similar scales
from sklearn.preprocessing import StandardScaler
#Using random forest classifier ensemble methos to increase accuracy
from sklearn.ensemble import RandomForestClassifier


def loadmodel(input_features):
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
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)
    #input_features = np.array([[sulphate, ph, solids, hardness,trihalomethanes, chloramines, conductivity, ocarbon, turbidity]])
    
    prediction = random_forest.predict(input_features)
    
    return prediction[0]





