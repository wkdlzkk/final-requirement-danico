import streamlit as st

# Set page title and subtitle
st.title("Sentiment Analyzer")
st.subheader("Enter your name and how you feel today")

# Input fields for name and message
name = st.text_input("What's your name?")
message = st.text_area("Tell me what you feel today")

# Lists of positive and negative words
positive_words = ['good', 'excited', 'happy', 'great', 'fantastic', 'wonderful']
negative_words = ['bad', 'sad', 'angry', 'terrible', 'awful', 'miserable']

# Function to analyze sentiment and display result
def sayFeeling():
    if name and message:
        st.markdown("---")  # Horizontal line for separation
        st.write(f"Hi, {name}!")

        # Convert message to lowercase and split into words
        words = message.lower().split()

        # Check for positive and negative words
        if any(word in positive_words for word in words):
            st.markdown("---")  # Horizontal line for separation
            st.write("That's good! :smile:")
        elif any(word in negative_words for word in words):
            st.markdown("---")  # Horizontal line for separation
            st.write("I hope you feel better soon. :disappointed:")
        else:
            st.markdown("---")  # Horizontal line for separation
            st.write("Keep going! :neutral_face:")
    else:
        st.warning("Please enter both your name and how you feel today.")

# Button to trigger sentiment analysis
st.button('Analyze Sentiment', on_click=sayFeeling)

# Additional content or features can be added below
