import pandas as pd
import streamlit as st 
from joblib import load

# Load the trained model
model = load('random_forest_model.joblib')

# Title and header for the Streamlit app
st.title('Laptop Price Prediction App')
st.header('Enter Laptop Information')

# Input fields for user input
company = st.selectbox('Company',('Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI', 'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer', 'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG'))
typename = st.selectbox('TypeName',('Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation'))
inches = st.number_input("Inches", min_value=0.0, max_value=100.0, value=1.37) 
ram = st.selectbox('Ram', (8, 16, 4, 2, 12, 6, 32, 24, 64))
memory = st.number_input('Memory', min_value=0, max_value=10000, value=128)
gpu = st.selectbox('Gpu', ('Intel', 'AMD', 'Nvidia'))
opsys = st.selectbox('OpSys', ('macOS', 'No OS', 'Windows 10', 'Linux', 'Chrome OS', 'Windows 7'))
touchscreen = st.selectbox('TouchScreen', ("Yes", "No"))
screen_height = st.number_input('Screen Height', min_value=0, max_value=10000, value=2560)
screen_weight = st.number_input('Screen Weight', min_value=0, max_value=10000, value=1600)
cpu_brand = st.selectbox('Cpu Brand', ('Intel', 'AMD'))
cpu_hz = st.number_input('Cpu HZ', min_value=0.0, max_value=10.0, value=2.3)

# Manually map categorical values to numeric codes based on the original LabelEncoder transformations
company_map = {'Apple': 0, 'HP': 1, 'Acer': 2, 'Asus': 3, 'Dell': 4, 'Lenovo': 5, 'Chuwi': 6, 'MSI': 7,
               'Microsoft': 8, 'Toshiba': 9, 'Huawei': 10, 'Xiaomi': 11, 'Vero': 12, 'Razer': 13, 'Mediacom': 14,'Samsung': 15, 'Google': 16, 'Fujitsu': 17, 'LG': 18}
typename_map = {'Ultrabook': 0, 'Notebook': 1, 'Gaming': 2, '2 in 1 Convertible': 3, 'Workstation': 4}
gpu_map = {'Intel': 0, 'AMD': 1, 'Nvidia': 2}
opsys_map = {'macOS': 0, 'No OS': 1, 'Windows 10': 2, 'Linux': 3, 'Chrome OS': 4, 'Windows 7': 5}
touchscreen_map = {'Yes': 1, 'No': 0}
cpu_brand_map = {'Intel': 0, 'AMD': 1}

# Convert categorical inputs to numeric values using the mappings
company_encoded = company_map[company]
typename_encoded = typename_map[typename]
gpu_encoded = gpu_map[gpu]
opsys_encoded = opsys_map[opsys]
touchscreen_encoded = touchscreen_map[touchscreen]
cpu_brand_encoded = cpu_brand_map[cpu_brand]

# Prepare the input data for prediction
input_features = [[company_encoded, typename_encoded, inches, ram, memory, gpu_encoded, opsys_encoded, touchscreen_encoded, screen_height, screen_weight, cpu_brand_encoded, cpu_hz]]

# Prediction
if st.button('Predict Price'):
    prediction = model.predict(input_features)
    st.success(f"The predicted price of the laptop is: ${prediction[0]:.2f}")
