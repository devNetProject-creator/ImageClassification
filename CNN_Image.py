import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sas
from  tensorflow.keras.layers import Dense,Conv2D
from tensorflow.keras.models import Sequential

# Data Preprocessing test
training_set = tf.keras.utils.image_dataset_from_directory(
    "E:/CNN_Image/Plants_2/valid",
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
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)
# Data Preprocessing test

validation_set = tf.keras.utils.image_dataset_from_directory(
    "E:/CNN_Image/Plants_2/test",
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
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True,
)
for x,y in training_set:
    print(x,x.shape)
    print(y,y.shape)
    break

model=Sequential()

#Building Convolution Layer

model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

# Dropout Layer
model.add(tf.keras.layers.Dropout(0.25))
#Flatten
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=1500,activation='relu'))
model.add(tf.keras.layers.Dropout(0.4)) #To avoid overfitting
#Output Layer
model.add(tf.keras.layers.Dense(units=22,activation='softmax'))
#Compiling and Training Phase
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
training_history = model.fit(x=training_set,validation_data=validation_set,epochs=10)

## Compiling and Training phase

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

training_history = model.fit(x=training_set,validation_data=validation_set,epochs=10)

#Output Layer
model.add(tf.keras.layers.Dense(units=22,activation='softmax'))
#Compiling and Training Phase
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
training_history = model.fit(x=training_set,validation_data=validation_set,epochs=10)

#Evaluating Model

#Training set Accuracy
train_loss, train_acc = model.evaluate(training_set)
print('Training accuracy:', train_acc)

#Validation set Accuracy
val_loss, val_acc = model.evaluate(validation_set)
print('Validation accuracy:', val_acc)

#Model Training
training_history = model.fit(x=training_set,validation_data=validation_set,epochs=10)

#Evaluating Model
#Training set Accuracy
train_loss, train_acc = model.evaluate(training_set)
print('Training accuracy:', train_acc)

#Validation set Accuracy
val_loss, val_acc = model.evaluate(validation_set)
print('Validation accuracy:', val_acc)

#Saving Model
model.save('trained_plant_disease_model.keras')
training_history.history #Return Dictionary of history

#Recording History in json
import json
with open('training_hist.json','w') as f:
  json.dump(training_history.history,f)
print(training_history.history.keys())

#Accuracy Visualization

epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='blue',label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.show()
import streamlit as st

