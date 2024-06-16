import streamlit as st
import pandas as pd
import pickle
from nltk.corpus import names

st.title("Sentiment Analyzer")
name = st.text_input("What's your name? ")
message = st.text_input("Tell me what you feel today: ")

positive_words = ['good', 'excited', 'happy', 'great', 'fantastic', 'wonderful']
negative_words = ['bad', 'sad', 'angry', 'terrible', 'awful', 'miserable']

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
