import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from xgboost import XGBRegressor


pipe = pickle.load(open('pipe.pkl', 'rb'))
data = pickle.load(open('laptop_data.pkl', 'rb'))


st.title("Laptop Predictor")

# Laptop Brand
Company = st.selectbox('Brand', data['Company'].unique())

# Laptop Type
TypeName = st.selectbox('Type', data['TypeName'].unique())

# Ram
Ram = st.selectbox('Ram in GB', data['Ram'].unique())

# Weight of Laptop
Weight = st.number_input('Weight of the Laptop')

# Touchscreen
Touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])

# IPS
IPS = st.selectbox('IPS', ['Yes', 'No'])

# Screen size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1440x900', '1920x1080', '1366x768', '1600x900', '3840x2160',
                                                '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# Cpu
Cpu_Brand = st.selectbox('Cpu', data['Cpu_Brand'].unique())

# HDD
HDD = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
SSD = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# Gpu
Gpu_Brand = st.selectbox('Gpu', data['Gpu_Brand'].unique())

# Operating System
OpSys = st.selectbox('OpSys', data['OpSys'].unique())



if st.button('Predict Price'):

    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0

    if IPS == 'Yes':
        IPS = 1
    else:
        IPS = 0

    resolution_width = int(resolution.split('x')[0])
    resolution_height = int(resolution.split('x')[1])

    PPI = ((resolution_width **2) + (resolution_height **2))**0.5 / screen_size


    query = np.array([Company, TypeName, Ram, OpSys, Weight, Touchscreen, IPS, PPI, Cpu_Brand, HDD, SSD, Gpu_Brand])

    result = query.reshape(1, 12)

    data = pd.DataFrame(result, columns = data.columns)

    st.title("The Predicted price of this configuration is Rs. " + str(np.round(np.exp(pipe.predict(data)[0]), 2)))
