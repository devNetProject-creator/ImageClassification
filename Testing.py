
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model, Sequential
from keras.layers import Resizing
from PIL import Image, ImageOps

validation_set = tf.keras.utils.image_dataset_from_directory(
    'E:/Git_CNN_Image/DevNetProject2/Plants_2/valid',
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
class_name = validation_set.class_names
         
print(class_name)

##Loading Model
cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')

##Visualising and Performing Prediction on Single image
#Test Image Visualization
import cv2
image_path = 'E:/Git_CNN_Image/DevNetProject2/Plants_2/test/Basil healthy (P8)/0008_0001.JPG'
# Reading an image in default mode
img = cv2.imread(image_path)
if img is None:
    print("Error: Image not loaded. Check the file path.")
else:
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Displaying the image 
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()
##Testing Model

image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = cnn.predict(input_arr)

print(predictions)

result_index = np.argmax(predictions) #Return index of max element
print(result_index)

# Displaying the disease prediction
model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()


