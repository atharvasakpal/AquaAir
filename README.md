<h1 align="center" id="title">AquaAir- AQI &amp; WQI Monitoring and Prediction</h1>

<p align="center"><img src="https://i.postimg.cc/L6XXBz7H/temp-Image-Y2-DSj8.avif" alt="project-image"></p>

<p id="description">Welcome the AquaAir your trusted source of real time AQI and WQI monitoring. Our mission is to empower you with the knowledge you need to make informed decisions about your health and well being.</p>

<h2>Project Screenshots:</h2>



<img src="https://i.postimg.cc/rmhpvJYF/temp-Imagehax-Pv-R.avif" alt="project-screenshot" width="1280" height="640/">

<img src="https://i.postimg.cc/g2Pg7FXs/temp-Imagefksmn-U.avif" alt="project-screenshot" width="1280" height="640/">

<img src="https://i.postimg.cc/mDtxrfSM/temp-Imagel-VVww-A.avif" alt="project-screenshot" width="1280" height="640/">

  
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   AquaAir AQI monitor displays the real time AQI value of any city you enter instantly. Along with the AQI values detailed interactive graphs of major pollutants with their composition is provided.
*   AquaAir uses XGBoost Linear Regression Machine Learning algorithm to read the historical data fed from our source api and to predict future values using Time Series Forecasting.

<img src="https://i.postimg.cc/FRytw5g5/temp-Imagefc-Qw-HO.avif" alt="project-screenshot" width="1280" height="640/">

  
*   AquaAir WQI monitor read values of different pollutant levels through a user input to then classify the sample as potable and unpotable.
*   The main web app runs on Django but an addition Streamlit backup is also created.

<h2>🛠️ Installation Steps:</h2>

<p>1. Enter the working directory for the django folder. In this case it is the src folder.</p>

<p>2. While inside the src folder open the terminal and type the following command:</p>

```
$ python .\manage.py runserver
```

<p>3. This command should return a link for the model which is running on a localhost.</p>

<p>4. In case some errors occur due to missing dependencies please install them on your local device using the pip install command.</p>

<p>5. To run the Streamlit version type the following command:</p>

```
streamlit run MainAPP.py
```

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Server: Django
*   Sckit-Learn
*   Streamlit
*   Plotly
