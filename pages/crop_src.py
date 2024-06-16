import streamlit as st


st.header('Crop Recommendation App')
st.subheader('This model was trained using a dataset')
st.code('''
       import streamlit as st
import pandas as pd
import pickle

# Load the trained Naive Bayes classifier from the saved file
filename = 'pages/crop_recom_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Function to predict crop based on input NPK levels
def predict_crop(n_input, p_input, k_input):
    if n_input == 0 and p_input == 0 and k_input == 0:
        return ""  # Return empty string if all inputs are zero
    else:
        # Predict using the loaded model
        crop_name = loaded_model.predict([[n_input, p_input, k_input]])
        return crop_name[0]  # Return the predicted crop name

# Streamlit app
st.title("Crop Predictor")
st.sidebar.subheader("Enter NPK levels:")

# Input sliders for NPK levels in the sidebar
n_input = st.sidebar.slider("Nitrogen", 0, 500)
p_input = st.sidebar.slider("Phosphorus", 0, 500)
k_input = st.sidebar.slider("Potassium", 0, 500)

# Predicting the crop based on NPK levels
crop_name = predict_crop(n_input, p_input, k_input)

# Display the predicted crop name in the main area
st.subheader("Predicted crop:")
st.write(crop_name)
    ''')
