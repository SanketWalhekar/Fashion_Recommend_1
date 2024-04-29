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

st.title('Fashion Recommendation System')
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

def recommend(features, feature_list, num_recommendations=6):
    neighbors = NearestNeighbors(n_neighbors=num_recommendations, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

def load_json_data(filename):
    json_filename = os.path.join('styles', os.path.splitext(filename)[0] + '.json')
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as json_file:
            return json.load(json_file)
    else:
        return None

def extract_product_info(json_data):
    product_info = {}
    if 'data' in json_data:
        data = json_data['data']
        if 'productDisplayName' in data:
            product_info['Name'] = data['productDisplayName']
        if 'price' in data:
            product_info['Price'] = data['price']
        if 'discountedPrice' in data:
            product_info['Discounted Price'] = data['discountedPrice']
        if 'brandName' in data:
            product_info['Brand'] = data['brandName']
        if 'landingPageUrl' in data:
            product_info['Website'] = data['landingPageUrl']
        if 'price' in data and 'discountedPrice' in data:
            discount_percentage = ((data['price'] - data['discountedPrice']) / data['price']) * 100
            product_info['Discount Percentage'] = f'{discount_percentage:.2f}%'
    return product_info

uploaded_file = st.file_uploader("Upload Your Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)
        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        # Recommend
        indices = recommend(features, feature_list)
        unique_indices = np.unique(indices)
        # Show recommendations
        if indices is not None and len(unique_indices) >= 2:
            st.subheader("Recommended Fashion Items:")
            for i in range(0, 6, 2):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Recommendation {i + 1}:")
                    st.image(filenames[unique_indices[i]], caption="Recommendation Image", use_column_width=True)
                    st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line for separation
                    # Load and display JSON data
                    json_data = load_json_data(os.path.basename(filenames[unique_indices[i]]))
                    if json_data:
                        product_info = extract_product_info(json_data)
                        st.write("Product Information:")
                        for key, value in product_info.items():
                            if key == 'Website':
                                st.markdown(f"[{value}]({value})", unsafe_allow_html=True)
                            else:
                                st.write(f"{key}: {value}")
                    else:
                        st.write("No product information available")
                with col2:
                    st.write(f"Recommendation {i + 2}:")
                    st.image(filenames[unique_indices[i + 1]], caption="Recommendation Image", use_column_width=True)
                    st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line for separation
                    # Load and display JSON data
                    json_data = load_json_data(os.path.basename(filenames[unique_indices[i + 1]]))
                    if json_data:
                        product_info = extract_product_info(json_data)
                        st.write("Product Information:")
                        for key, value in product_info.items():
                            if key == 'Website':
                                st.markdown(f"[{value}]({value})", unsafe_allow_html=True)
                            else:
                                st.write(f"{key}: {value}")
                    else:
                        st.write("No product information available")
        else:
            st.error("No recommendations available for the uploaded image.")
    else:
        st.error("An error occurred during file upload. Please try again.")
