import streamlit as st
import pandas as pd
import numpy as np
import time
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from Homepage import display
from WQI_display_prediction import loadmodel
from AQI_display_prediction import show_aqi, load_dataset, healthyBarPlot, healthyPiePlot, pastPlot,mlModel


 


st.set_page_config(
    page_title = 'AquaAir- AQI & WQI Monitoring and Prediction'
)

## hiding streamlit watermark
hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def aqi_page(city): 
    if city != '':
        aqi = show_aqi(city)
        st.write('## The AQI of {} is {}'.format(city,aqi))




with st.sidebar:
    selected = option_menu(options=['Home','AQI','WQI','About'],
                           menu_title = None,
                           
                           )
    
if selected == 'Home':
    display()



if selected == 'AQI':
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.title('Air Quality Index')
    city = st.text_input('Enter City')
    aqi_page(city)
    if city != '':
        st.write('### Showing the detailed info about air quality')
        daily_input = load_dataset(city)
        
        figBar = healthyBarPlot(daily_input)
        st.write(figBar)
        figPie = healthyPiePlot(daily_input)
        st.write(figPie)

        st.write('### Showing Historical Data and Predicting future plots')
        fig = pastPlot(daily_input)
        st.write(fig)
        figPred = mlModel(daily_input)
        st.write(figPred)
        st.write('_The data of pm2.5, pm10, no2, so2 is fetched from OpenWeather Air Pollution API_')




if selected == 'WQI':
    st.title(f'You have selected {selected}')    
    st.write("Our WQI model inputs concentrations of various contaminants to detect the given sample's potability")
    
    
    with st.form('My form'):
        st.write('### Input Values: ')
        sulphate = st.number_input('Enter Sulphates')
        ph = st.number_input('Enter ph')
        solids = st.number_input('Enter Solids')
        hardness = st.number_input('Enter Hardness')
        trihalomethanes = st.number_input('Enter Trihalomethanes')
        chloramines = st.number_input('Enter Chloramines')
        conductivity = st.number_input('Enter Conductivity')
        ocarbon = st.number_input('Enter Organic Carbon') 
        turbidity = st.number_input('Enter Turbidity')
        submitted = st.form_submit_button('SUBMIT')
        if submitted:
            input_features = np.array([[sulphate, ph, solids, hardness,trihalomethanes, chloramines, conductivity, ocarbon, turbidity]])
            WQImodel= loadmodel(input_features)
            
            if (WQImodel ==1):
                st.write("### The given sample is Potable !!")
            else:
                st.write("### The given sample is not Potable !!") 
    st.write('_The training dataset for RandomForest Classifier model is fetched from Kaggle_')           

    





if selected == 'About':
    st.title('About')
    st.write('## Problem Statement')
    st.write('Considering the importance of air and water to human existence, air pollution and water pollution are critical issues that require collective effort for prevention and control. Different types of anthropogenic activities have resulted in environmental dilapidation and ruin. One of the tools that can be used for such a campaign is Air Quality Index (AQI). The AQI was based on the concentrations of different pollutants: We are also familiar with the Water Quality Index (WQI), which in simple terms tells what the quality of drinking water is from a drinking water supply. There is a need for constant and continuous environment monitoring of air quality and water quality for the development of AQI and WQI, which in turn will enable clear communication of how clean or unhealthy the air and water in the study area is.')
    st.write('## Contributors')
    st.write('[Atharva Sakpal](https://github.com/CaptainAtharva)')
    st.write('[Anish Mahadevan](https://github.com/Faulty404)')
    st.write('[Shreya Chaudhari](https://github.com/passthatzaza)')
    st.write('[Ishayu Potey](https://github.com/ISH2YU)')
    st.write('[Keyur Apte](https://github.com/Keyamy100)')
    st.write('[Shashwat Barai](https://github.com/Shash2106)')
    st.write('')
    st.write('## Future updates')
    st.write('-Work on implementation of better framework for the project')
    st.write('-The ML model accuracy stands  at 75%. Working to increase the accuracy of the model.')
    st.write('-Improvement on better resoponse time for data collection and processing from the OpenWeather API')
    st.write('-The WQI model relies on manually entered data. Working to find a suitable API to get concentration values of a city directly.')

