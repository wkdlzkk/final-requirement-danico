import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title, hide_pages

add_page_title()
show_pages(
    [   
        Section("Vernie Jaica M. Danico BSIS3C"),
        Page("home.py",),
        Section("Applications"),
        Page("pages/crop_recom_streamlitapp.py", "Prediction", "1️⃣", in_section=True),
        Page("pages/basic_sentiment_analyzer.py", "Sentiment Analysis", "2️⃣", in_section=True),
        Page("pages/activity_endterm_4.py", "Image Classification", "3️⃣", in_section=True),


        Section("Source Codes"),
        Page("pages/crop_src.py", "Prediction Source", "1️⃣", in_section=True),
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




st.markdown("""

--------------------------   
""", unsafe_allow_html=True)


st.markdown("""

-----------------------------

""", unsafe_allow_html=True)


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
