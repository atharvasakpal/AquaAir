import streamlit as st
def display():
    st.write('# AquaAir')
    st.write('### AQI & WQI Monitoring and Prediction')
    st.write('Welcome the AquaAir, your trusted source of real time AQI and WQI monitoring. Our mission is to empower you with the knowledge you need to make informed decisions about your health and well being.')
    st.write('AquaAir AQI monitor displays the real time AQI value of any city you enter instantly. Along with the AQI values, detailed interactive graphs of major pollutants with their composition is provided.')
    st.write('AquaAir uses XGBoost Linear Regression Machine Learning algorithm to read the historical data fed from our source api and to predict future values using Time Series Forecasting.')
    st.write('AquaAir WQI monitor read values of different pollutant levels through a user input, to then classify the sample as potable and unpotable.')
    st.write('AquaAir WQI potability ML model uses RandomForest classification technique.')
    st.write('')
    st.write('Click on the different sections on the sidebar explore our WebApp.')
