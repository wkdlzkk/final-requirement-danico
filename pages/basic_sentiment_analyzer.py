import streamlit as st

st.title("Sentiment Analyzer")
name = st.text_input("What's your name?")
message = st.text_input("Tell me what you feel today:")

positive_words = ['good', 'excited', 'happy', 'great', 'fantastic', 'wonderful']
negative_words = ['bad', 'sad', 'angry', 'terrible', 'awful', 'miserable']

def sayFeeling():
    if name and message:
        st.write(f"Hi, {name}!")
        words = message.lower().split()
        if any(word in positive_words for word in words):
            st.write("That's good! :smile:")
        elif any(word in negative_words for word in words):
            st.write("I hope you feel better soon. :disappointed:")
        else:
            st.write("Keep going! :neutral_face:")
    else:
        st.write("Please enter both your name and how you feel today.")

st.button('Say it', on_click=sayFeeling)
