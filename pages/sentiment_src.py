import streamlit as st


st.header('Simple Sentiment Analyzer App')
st.subheader('This python code is implemented for Streamlit')
st.code('''
       import streamlit as st
import pandas as pd
import pickle
from nltk.corpus import names

st.title("Sentiment Analyzer")
name = st.text_input("What's your name? ")
message = st.text_input("Tell me what you feel today: ")

# Define lists of positive and negative words
positive_words = ['good', 'excited', 'happy', 'great', 'fantastic', 'wonderful']
negative_words = ['bad', 'sad', 'angry', 'terrible', 'awful', 'miserable']

# Function to classify the sentiment and display a message
def sayFeeling():
    st.write(f"Hi, {name}!")
    words = message.lower().split()
    if any(word in positive_words for word in words):
        st.write("That's good! :smile:")
    elif any(word in negative_words for word in words):
        st.write("I hope you feel better soon. :disappointed:")
    else:
        st.write("Keep going! :neutral_face:")

st.button('Say it', on_click=sayFeeling)

# To run on terminal issue this command

