import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title, hide_pages

add_page_title()
show_pages(
    [   
        Page("home.py",),
        Section("Applications"),
        Page("pages/crop_recom_streamlitapp.py", "Prediction", "1️⃣", in_section=True),
        Page("pages/basic_sentiment_analyzer.py", "Sentiment Analysis", "2️⃣", in_section=True),
        Page("pages/activity_endterm_4.py", "Image Classification", "3️⃣", in_section=True),


        Section("Source Codes"),
        Page("pages/sentiment_src.py", "Sentiment Analysis Source", "2️⃣", in_section=True),
        Page("pages/image_classification_src.py", "Image Classification Source", "3️⃣", in_section=True),
]
)



st.markdown("""

##### 👨‍🎓About Me:
At 22, I'm a student at Carlos Hilado Memorial State University, 
studying for a Bachelor of Science in Information Systems. I chose 
this path because I've always loved tinkering with computers. From 
coding to problem-solving, I find it all fascinating. As I progress 
through my studies, I'm excited to learn more about databases, 
programming languages, and network systems. I believe that with 
dedication and a love for learning, I can turn my passion for technology 
into a rewarding career.          

### 💾 Machine Learning

##### 💻 Applications

* Prediction
* Sentiment Analysis
* Image Classification



### 🔎 Overview""", unsafe_allow_html=True)


st.image("./sentiment_analyzer.png")


st.markdown("""
This application is designed to analyze the sentiment expressed in text messages. It utilizes a trained Naive Bayes classifier to classify text into positive, negative, or neutral sentiments based on the words used.

How it Works:

Input: Users input their name and describe how they are feeling today in the provided text boxes.
Analysis: The application examines the words in the input message to determine its sentiment.
Classification: Words are compared against predefined lists of positive, negative, and neutral words. If a match is found, the sentiment is displayed accordingly.
Feedback: Additionally, users receive motivational messages tailored to their sentiment to uplift their mood.
Output: The application outputs the user's name along with an assessment of their sentiment and an encouraging message.
Key Features:

Customizable: Users can input their name and express their feelings freely.
Real-time Analysis: Sentiment analysis is performed instantly upon clicking the "Analyze Sentiment" button.
Feedback: Users receive personalized messages to help them feel better based on their sentiment.
Try it Out:

Enter your name.
Describe how you are feeling today.
Click the "Analyze Sentiment" button to receive feedback.
Feel free to express yourself and see how the Sentiment Analyzer responds!


            
### ⭐ Star the project on Github  <iframe src="https://ghbtns.com/github-btn.html?user=koalatech&repo=streamlit_web_app&type=star&count=true"  width="150" height="20" title="GitHub"></iframe>   
""", unsafe_allow_html=True)

st.image("./food_prediction.png")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
