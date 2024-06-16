import streamlit as st
import pickle

# Load the trained Naive Bayes classifier from the saved file
filename = 'pages/crop_recom_model.sav'

try:
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file '{filename}' not found. Please make sure the file path is correct.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to predict crop based on input NPK levels
def predict_crop(n_input, p_input, k_input):
    if n_input == '' or p_input == '' or k_input == '':
        return "Enter Nitrogen, Phosphorus, and Potassium:"  
    else:
        try:
            # Predict using the loaded model
            n = float(n_input)
            p = float(p_input)
            k = float(k_input)
            crop_name = loaded_model.predict([[n, p, k]])
            return crop_name[0]  # Return the predicted crop name
        except Exception as e:
            return f"Prediction error: {e}"

# Streamlit app
st.title("Crop Predictor")

# Input fields for NPK levels
n_input = st.text_input("Nitrogen", "")
p_input = st.text_input("Phosphorus", "")
k_input = st.text_input("Potassium", "")

# Predicting the crop
crop_name = predict_crop(n_input, p_input, k_input)

# Display the predicted crop name
st.text("Predicted crop based on NPK levels:")
st.text(crop_name)
