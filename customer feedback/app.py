import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and preprocess the dataset
def load_data():
    df = pd.read_csv('./customer_feedback_satisfaction.csv')
    df.drop(columns=['CustomerID'], inplace=True)
    df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 2})
    df['Country'] = df['Country'].replace({'UK': 1, 'USA': 2, 'France': 3, 'Germany': 4, 'Canada': 5})
    df['FeedbackScore'] = df['FeedbackScore'].replace({'Low': 1, 'Medium': 2, 'High': 3})
    df['LoyaltyLevel'] = df['LoyaltyLevel'].replace({'Bronze': 1, 'Gold': 2, 'Silver': 3})
    
    return df

# Load and prepare the data
df = load_data()

# Define feature columns (ensure this matches the columns used for training)
feature_columns = ['Gender', 'Country', 'ProductQuality', 'ServiceQuality', 'FeedbackScore', 'LoyaltyLevel', 'Age', 'Income', 'PurchaseFrequency']

# Splitting the data into features and target variable
X = df[feature_columns]
y = df['SatisfactionScore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=10, min_samples_split=8)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title('Customer Feedback Satisfaction Prediction')

st.write("""
    This app predicts customer satisfaction based on various features such as gender, country, product quality, and feedback score.
""")

# Show Feature Importance
st.subheader('Feature Importance')
feature_importances = model.feature_importances_
features_df = pd.DataFrame({'Feature': feature_columns, 'Importance': feature_importances})
features_df = features_df.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=features_df, ax=ax)
st.pyplot(fig)

# User input form for predictions
st.sidebar.header('User Input')

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
country = st.sidebar.selectbox("Country", ["UK", "USA", "France", "Germany", "Canada"])
product_quality = st.sidebar.slider("Product Quality", 1, 5)
service_quality = st.sidebar.slider("Service Quality", 1, 5)
feedback_score = st.sidebar.selectbox("Feedback Score", ["Low", "Medium", "High"])
loyalty_level = st.sidebar.selectbox("Loyalty Level", ["Bronze", "Gold", "Silver"])

# Additional user inputs for missing features
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Income", min_value=1000, max_value=200000, value=50000)
purchase_frequency = st.sidebar.slider("Purchase Frequency", 1, 12, value=5)

# Create a DataFrame for user input
user_input = pd.DataFrame({
    'Gender': [1 if gender == "Male" else 2],
    'Country': [{"UK": 1, "USA": 2, "France": 3, "Germany": 4, "Canada": 5}[country]],
    'ProductQuality': [product_quality],
    'ServiceQuality': [service_quality],
    'FeedbackScore': [{"Low": 1, "Medium": 2, "High": 3}[feedback_score]],
    'LoyaltyLevel': [{"Bronze": 1, "Gold": 2, "Silver": 3}[loyalty_level]],
    'Age': [age],
    'Income': [income],
    'PurchaseFrequency': [purchase_frequency]
}, columns=feature_columns)

# Ensure the order of features in user input matches the model's training order
user_input = user_input[feature_columns]

# Prediction triggered by button
if st.sidebar.button('Predict'):
    # Make the prediction
    prediction = model.predict(user_input)
    st.write(f"**Predicted Satisfaction Score:** {prediction[0]:.2f}")
else:
    st.write("Click 'Predict' to get the Satisfaction Score prediction.")