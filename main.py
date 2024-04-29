import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import json

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Load feature embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')
st.write("Welcome to the Fashion Recommender System. Upload an image to find similar fashion items.")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(244, 244))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

uploaded_file = st.file_uploader("Upload Your Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)
        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        # Recommend
        indices = recommend(features, feature_list)
        # Show recommendations
        if indices is not None and len(indices[0]) >= 6:
            st.subheader("Recommended Fashion Items:")
            st.write("Here are some fashion items similar to the uploaded image:")
            for i in range(0, 6, 2):
                row = st.columns(2)
                with row[0]:
                    st.image(filenames[indices[0][i]], caption="Recommendation {}".format(i + 1), use_column_width=True)
                with row[1]:
                    st.image(filenames[indices[0][i + 1]], caption="Recommendation {}".format(i + 2), use_column_width=True)
        else:
            st.error("No recommendations available for the uploaded image.")
    else:
        st.error("An error occurred during file upload. Please try again.")
