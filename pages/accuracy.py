# Notes
# do a "pip install streamlit" first 
# to run on terminal issue this command
# python -m streamlit run streamlit_test.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle
import os

# Check current directory
print(f"Current directory: {os.getcwd()}")

# Load the trained Random Forest classifier from the saved file
model_path = 'prediction.p'

# Check if file exists
if os.path.exists(model_path):
    print(f"Model file found at: {model_path}")
else:
    print(f"Model file not found at: {model_path}")

try:
    loaded_model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model: {e}")

# Load your dataset and perform necessary preprocessing
@st.cache
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Check unique values in the 'brand' column
    print(f"Unique brands: {df['brand'].unique()}")

    # Data preprocessing steps
    X = df[['Price', 'Rating']]
    y = df['year_of_warranty']

    # Ensure 'brand' column is treated as categorical
    df['brand'] = df['brand'].astype('category')

    brand_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    brand_encoded = brand_encoder.fit_transform(df[['brand']])
    
    # Verify encoded categories
    print(f"Encoded categories: {brand_encoder.categories_}")

    X = pd.concat([pd.DataFrame(brand_encoded), X], axis=1)
    X.columns = X.columns.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, brand_encoder


# Function to make predictions
def make_prediction(model, brand_encoder, new_instance):
    new_instance_brand_encoded = brand_encoder.transform(pd.DataFrame(new_instance['brand']))
    new_instance_encoded = pd.concat([pd.DataFrame(new_instance_brand_encoded), pd.DataFrame(new_instance[['Price', 'Rating']])], axis=1)
    new_instance_encoded.columns = new_instance_encoded.columns.astype(str)
    prediction = model.predict(new_instance_encoded)
    return prediction

# Streamlit app
def main():
    st.title('Laptop Warranty Prediction')
    st.sidebar.header('New Laptop Instance')

    # Sidebar inputs for new instance
    price = st.sidebar.number_input('Price', min_value=0, step=100)
    rating = st.sidebar.number_input('Rating', min_value=0.0, max_value=5.0, step=0.1)
    brand = st.sidebar.selectbox('Brand', ['asus', 'dell', 'hp', 'lenovo', 'acer', 'msi', 'apple'])

    new_instance = {
        'Price': [price],
        'Rating': [rating],
        'brand': [brand]
    }

    # Load and preprocess data
    file_path = 'pages/laptops.csv'  # Update with your dataset path
    X_train, X_test, y_train, y_test, brand_encoder = load_and_preprocess_data(file_path)

    # Make predictions
    try:
        prediction = make_prediction(loaded_model, brand_encoder, new_instance)
        if st.sidebar.button('Predict Warranty'):
            if prediction[0] == 1:
                st.write("Model predicts the warranty to be of year 2 or more.")
            else:
                st.write("Model predicts the warranty to be of year less than 2.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

if __name__ == '__main__':
    main()
