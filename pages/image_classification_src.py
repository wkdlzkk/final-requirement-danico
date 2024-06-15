import streamlit as st


st.header('Image Classification App')
st.subheader('This python code is implemented for Streamlit')
st.code('''
        import pickle
from PIL import Image
from io import BytesIO
from img2vec_pytorch import Img2Vec
import streamlit as st

st.set_page_config(layout="wide", page_title="Image Classification for Foods")

st.write("## Image Classification Model for Foods")
st.write(":grin: Predicting food categories from uploaded images :grin:")
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Load the pre-trained model
with open('/content/foods.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

@st.cache
def classify_image(upload):
    image = Image.open(upload)
    st.write("### Image to be predicted:")
    st.image(image)

    st.write("### Predicted Category:")
    features = img2vec.get_vec(image)
    pred = model.predict([features])

    st.header(pred[0])  # Display the predicted category

col1, col2 = st.columns(2)
my_upload = col1.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        classify_image(my_upload)
else:
    st.write("Upload an image to classify it.")
    ''')
