import streamlit as st
from keras.models import load_model, Sequential
from keras.layers import Resizing
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model with resizing layer for input compatibility

test_set = tf.keras.utils.image_dataset_from_directory(
    'E:/Git_CNN_Image/DevNetProject2/Plants_2/test',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
class_name = test_set.class_names

@st.cache_resource  # Cache the model so it doesn't reload every time the script runs
def load_keras_model():
    base_model = load_model("E:/Git_CNN_Image/DevNetProject2/trained_plant_disease_model.keras")
    model = Sequential([Resizing(128, 128), base_model])  # Adjust dimensions to (128, 128)
    return model

model = load_keras_model()


# Function to preprocess the image for the model
def preprocess_image(image):
    image = ImageOps.fit(image, (128, 128), Image.LANCZOS)  # Resize image to 128x128
    image = np.asarray(image)
    image = image / 255.0  # Normalize if your model was trained on normalized images
    image = np.expand_dims(image, axis=0)  # Model expects a batch of images
    return image

# Streamlit interface
st.title("Plant Health Prediction")
st.write("Upload an image of the plant, and the model will predict whether it's healthy or not.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process and predict on submit
    if st.button("Submit"):
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        print(prediction)
        # Get the predicted class label
        result_index = np.argmax(prediction)  # Return index of max element
        model_prediction = class_name[result_index]
       
        # Display the result
        st.write(f"Disease Name: {model_prediction}")
        
        # Plot and display the image with the title as prediction
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title(f"Disease Name: {model_prediction}")
        ax.axis("off")
        st.pyplot(fig)
