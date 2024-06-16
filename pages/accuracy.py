# Notes
# do a "pip install streamlit" first 
# to run on terminal issue this command
# python -m streamlit run streamlit_test.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
import pickle

# Load your dataset and perform necessary preprocessing
@st.cache
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Data preprocessing steps (similar to your previous code)
    X = df[['Price', 'Rating']]
    y = df['year_of_warranty']

    brand_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    brand_encoded = brand_encoder.fit_transform(df[['brand']])
    X = pd.concat([pd.DataFrame(brand_encoded), X], axis=1)
    X.columns = X.columns.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, brand_encoder

# Train and save your model
@st.cache(allow_output_mutation=True)
def train_and_save_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model using pickle
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

# Load the saved model
def load_saved_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to make predictions
def make_predictions(model, scaler, brand_encoder, new_instance):
    new_instance_brand_encoded = brand_encoder.transform(pd.DataFrame(new_instance['brand']))
    new_instance_encoded = pd.concat([pd.DataFrame(new_instance_brand_encoded), pd.DataFrame(new_instance[['Price', 'Rating']])], axis=1)
    new_instance_encoded.columns = new_instance_encoded.columns.astype(str)
    new_instance_scaled = scaler.transform(new_instance_encoded)
    prediction = model.predict(new_instance_scaled)
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
    file_path = '/content/drive/MyDrive/Dataset/laptops.csv'  # Update with your file path
    X_train, X_test, y_train, y_test, brand_encoder = load_and_preprocess_data(file_path)

    # Train and save model
    model = train_and_save_model(X_train, y_train)

    # Load saved model
    model_path = 'random_forest_model.pkl'
    loaded_model = load_saved_model(model_path)

    # Make predictions
    prediction = make_predictions(loaded_model, X_train, brand_encoder, new_instance)

    if st.sidebar.button('Predict Warranty'):
        if prediction[0] == 1:
            st.write("Model predicts the warranty to be of year 2 or more.")
        else:
            st.write("Model predicts the warranty to be of year less than 2.")

if __name__ == '__main__':
    main()
