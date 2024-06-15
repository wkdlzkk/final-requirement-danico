
!pip install img2vec_pytorch
!pip install scikit-learn==1.4.2

from google.colab import drive
drive.mount('/content/drive')

import os
import pickle
import numpy as np

from img2vec_pytorch import Img2Vec
from PIL import Image, UnidentifiedImageError
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

img2vec = Img2Vec()

data_dir = '/content/drive/MyDrive/dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')



data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)

            img = Image.open(img_path_).convert('RGB')

            img_features = img2vec.get_vec(img)

            features.append(img_features)
            labels.append(category)

    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels

from matplotlib import pyplot as plt
import os

data_dir = '/content/drive/MyDrive/dataset/train'

class_names = sorted(os.listdir(data_dir))
nrows = len(class_names)
ncols = 5
plt.figure(figsize=(ncols*1.5, nrows*1.5))

for row in range(nrows):
    class_name = class_names[row]
    img_paths = [os.path.join(data_dir, class_name, filename)
                 for filename in os.listdir(os.path.join(data_dir, class_name))]
    for col in range(min(ncols, len(img_paths))):
        plt.subplot(nrows, ncols, row*ncols + col + 1)
        img = plt.imread(img_paths[col])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(class_name, fontsize=8)
plt.tight_layout()
plt.show()

print(data.keys())

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
     'kernel': ['rbf', 'poly'],

}


model = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2,scoring='accuracy' )

model.fit(data['training_data'], data['training_labels'])



print(model.best_params_)
print(model.best_score_)

# save the model
with open('/content/foods.p', 'wb') as f:
    pickle.dump(model, f)
    f.close()

import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image



with open('/content/foods.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()
class_labels = ['-K', '-N', '-P', 'FN']
image_path = '/content/drive/MyDrive/dataset/train/pizza/pizza3.jpg'

img = Image.open(image_path)

features = img2vec.get_vec(img)
features_2d = features.reshape(1, -1)

# Get prediction probabilities
prediction_probabilities = model.predict_proba(features_2d)[0]
for ind, prob in enumerate(prediction_probabilities):
    print(f'Class {class_labels[ind]}: {prob*100:.2f}%')

pred = model.predict([features])

print(pred)

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import pickle
# from PIL import Image
# from io import BytesIO
# from img2vec_pytorch import Img2Vec
# import streamlit as st
# 
# with open('foods.p', 'rb') as f:
#     model = pickle.load(f)
# 
# img2vec = Img2Vec()
# 
# ## Streamlit Web App Interface
# st.set_page_config(layout="wide", page_title="Image Classification for Foods")
# 
# st.write("## Image Classification Model in Python!")
# st.write(
#     ":grin: We'll try to predict the image on what features it was trained via the uploaded image :grin:"
# )
# st.sidebar.write("## Upload and download :gear:")
# 
# MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
# 
# # Download the fixed image
# @st.cache_data
# def convert_image(img):
#     buf = BytesIO()
#     img.save(buf, format="jpg")
#     byte_im = buf.getvalue()
#     return byte_im
# 
# def fix_image(upload):
#     image = Image.open(upload)
#     col1.write("Image to be predicted : camera :")
#     col1.image(image)
# 
#     col2.write("Category : foods :")
#     img = Image.open(my_upload)
#     features = img2vec.get_vec(img)
#     pred = model.predict([features])
# 
#     col2.header(pred)
# 
# 
# 
# col1, col2 = st.columns(2)
# my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
# 
# if my_upload is not None:
#     if my_upload.size > MAX_FILE_SIZE:
#         st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
#     else:
#         fix_image(upload=my_upload)
# else:
#     st.write("by Vernie Jaica Danico...")

! wget -q -O - ipv4.icanhazip.com

! streamlit run app.py & npx localtunnel --port 8501
