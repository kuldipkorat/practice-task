import streamlit as st
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

# Load the trained Random Forest model
model = load('gr.pkl')

# Streamlit App Title with Emoji
st.title("Gym Member Exercise Tracker - Calories Burned Predictor")


# Collect User Inputs for Prediction
st.header("Enter Your Exercise Details:")
age = st.number_input("Age (in years)", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ("Male", "Female"))
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7)
max_bpm = st.number_input("Max BPM", min_value=100, max_value=220, value=180)
avg_bpm = st.number_input("Avg BPM", min_value=50, max_value=200, value=130)
resting_bpm = st.number_input("Resting BPM", min_value=40, max_value=100, value=60)
session_duration = st.number_input("Session Duration (hours)", min_value=0.1, max_value=5.0, value=1.0)
workout_type = st.selectbox("Workout Type", ("Yoga", "HIIT", "Cardio", "Strength"))
fat_percentage = st.number_input("Fat Percentage", min_value=5.0, max_value=50.0, value=20.0)
water_intake = st.number_input("Water Intake (liters)", min_value=0.5, max_value=5.0, value=2.0)
workout_frequency = st.number_input("Workout Frequency (days/week)", min_value=1, max_value=7, value=3)
experience_level = st.selectbox("Experience Level", ("1", "2", "3"))
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

# Encode categorical inputs
gender = 1 if gender == "Male" else 0
workout_type_map = {"Yoga": 0, "HIIT": 1, "Cardio": 2, "Strength": 3}
workout_type = workout_type_map[workout_type]
experience_level = int(experience_level)

# Prepare input data for prediction
input_data = np.array([[age, gender, weight, height, max_bpm, avg_bpm, resting_bpm, session_duration,
                        workout_type, fat_percentage, water_intake, workout_frequency, experience_level, bmi]])

# Button for making prediction
if st.button("Predict Calories Burned"):
    prediction = model.predict(input_data)
    
    # Displaying result in a unique way
    st.markdown(f"### 🔥 Predicted Calories Burned: **{prediction[0]:.2f} kcal**")
    
    # Create a simple bar chart visualizing input parameters and their effects on calories burned
    feature_names = ['Age', 'Weight', 'Height', 'Max BPM', 'Avg BPM', 'Session Duration']
    feature_values = [age, weight, height, max_bpm, avg_bpm, session_duration]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feature_names, feature_values, color='skyblue')
    ax.set_xlabel("Values")
    ax.set_title("Feature Impact on Calories Burned Prediction")
    
    st.pyplot(fig)

    # Bonus: Suggest improvements for users
    if prediction[0] > 500:
        st.success("Great workout! You're burning a lot of calories!")
    else:
        st.info("Keep pushing! Try increasing your workout intensity to burn more calories.")