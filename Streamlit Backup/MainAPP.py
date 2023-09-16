import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from Homepage import display
from AQI_display_prediction import show_aqi, load_dataset, healthyBarPlot, healthyPiePlot, pastPlot,mlModel
st.set_page_config(
    page_title = 'AquaAir- AQI & WQI Monitoring and Prediction'
)

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
    st.write('[Shashwat Barai](https://github.com/)')
    st.write('')
    st.write('## Future updates')
    st.write('-Work on implementation of better framework for the project')
    st.write('-The ML model accuracy stands  at 75%. Working to increase the accuracy of the model.')