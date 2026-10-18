import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('heart_model.pkl')

st.title("Heart Disease Prediction System")
st.write("Enter patient details to predict heart disease risk.")

# Input fields for your 4 attributes
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)

if st.button("Predict"):
    input_data = np.array([[age, sex, trestbps, chol]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("Prediction: High risk of heart disease.")
    else:
        st.success("Prediction: Low risk of heart disease.")