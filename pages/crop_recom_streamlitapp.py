#Notes
# do a "pip install streamlit" first 
#to run on terminal issue this command
# python -m streamlit run streamlit_test.py

import streamlit as st
import pandas as pd
import pickle
from nltk.corpus import names

# Load the trained Naive Bayes classifier from the saved file
filename = 'pages/crop_recom_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# # Use the model to make predictions
@st.cache_data 
def predict_crop():
    st.text("The crop is " + crop_name)
    return
           
st.title("Crop Predictor")
st.subheader("Enter NPK:")
n_input = st.slider("Nitrogen: ",0,500)
p_input = st.slider("Phosphorus: ",0,500)
k_input = st.slider("Potassium: ",0,500)
if n_input == 0 & p_input == 0 & k_input == 0:
    crop_name = ""
else:
    crop_name = loaded_model.predict([[pd.to_numeric(n_input),pd.to_numeric(p_input),pd.to_numeric(k_input)]])

st.text("Predicted crop:")
st.text_area(label ="",value=crop_name, height =100)
# st.button('Predict', on_click=predict_crop
