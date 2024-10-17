import streamlit as st
from siamese_network import build_siamese_network, compare_images
from PIL import Image
import numpy as np

st.title('Dog Image Comparison using Siamese Network')

# Create an image uploader
uploaded_file_1 = st.file_uploader("Choose the first dog image", type=["jpg", "png"])
uploaded_file_2 = st.file_uploader("Choose the second dog image", type=["jpg", "png"])

# If both images are uploaded, display them
if uploaded_file_1 and uploaded_file_2:
    img1 = Image.open(uploaded_file_1)
    img2 = Image.open(uploaded_file_2)

    st.image(img1, caption='First Image', use_column_width=True)
    st.image(img2, caption='Second Image', use_column_width=True)

    # Preprocess images
    img1_array = np.array(img1.resize((224, 224))) / 255.0
    img2_array = np.array(img2.resize((224, 224))) / 255.0

    # Build and load the Siamese model
    input_shape = (224, 224, 3)
    model = build_siamese_network(input_shape)

    # Compare the images
    similarity_score = compare_images(model, img1_array, img2_array)

    st.write(f"Similarity Score: {similarity_score}")
