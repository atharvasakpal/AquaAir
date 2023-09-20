#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Scaling data to ensure similar scales
from sklearn.preprocessing import StandardScaler

#GridSearchCV for hyperparameter tuning
#from sklearn.model_selection import GridSearchCV

#Using random forest classifier ensemble methos to increase accuracy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
model.fit(X_train,y_train)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)


# In[4]:


city = input("Enter City name : ")

Sulfate = float(input("Sulfate in the water: "))
ph = float(input("ph of water: "))
Solids = float(input("Solids in the water: "))
Hardness = float(input("Hardness of the water: "))
Trihalomethanes = float(input("Trihalomethanes in the water: "))
Chloramines = float(input("Chloramines in the water: "))
Conductivity = float(input("Conductivity of the water: "))
Organic_carbon= float(input("Organic_carbon in the water: "))
Turbidity = float(input("Turbidity of the water: "))

input_features = np.array([[Sulfate, ph, Solids, Hardness,Trihalomethanes, Chloramines, Conductivity, Organic_carbon, Turbidity]])

prediction = random_forest.predict(input_features)
if(prediction[0] == 0):
    print(f"Water is not potable: {prediction[0]}")
else:
    print(f"Water is potable: {prediction[0]}")


# In[ ]:




