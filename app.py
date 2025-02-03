import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.joblib')

# Create a Streamlit app
st.title('Iris Species Prediction')

# Input fields for the iris dataset features
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, value=1.0)

# Predict the class for the input features
input_features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                              columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
prediction = model.predict(input_features)[0]

# Display the predicted class
st.write(f'The predicted class is: {prediction}')
