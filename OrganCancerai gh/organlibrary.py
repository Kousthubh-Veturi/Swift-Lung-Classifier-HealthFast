import urllib
from urllib import request
import pathlib 
from pathlib import Path
import shutil
import os
from numpy import save
import pathlib
import os
from pathlib import Path
import tensorflow as tf
import keras
from keras import models 
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import resnet
from keras.models import Model
from keras.applications import efficientnet_v2
from keras.applications import nasnet
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from keras.applications import inception_v3
import pickle as pkl
import glob
from typing import Any
import copy
from copy import deepcopy
from keras.preprocessing.image import ImageDataGenerator

directory = "lung_image_sets"
image_dim = (224,224)
classes = 3
ttsplit=0.8


#augmentation
trainaug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
    )
testaug = ImageDataGenerator(rescale=1./255)


#split and gen
traindiv = trainaug.flow_from_directory(
    "lung_image_sets",
    target_size=image_dim,
    batch_size=128,
    class_mode='categorical',
    subset='training',
    shuffle=True
    )
testdiv = trainaug.flow_from_directory(
    "lung_image_sets",
    target_size=image_dim,
    batch_size=128,
    class_mode='categorical',
    subset='validation',
    shuffle=False
    )


#labels assignign
print(traindiv.class_indices)
print("Training set size:", traindiv.samples)
print("Validation set size:", testdiv.samples)


#compilification
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
   #tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
   #tf.keras.layers.MaxPooling2D(2, 2),
   #tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
   #tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(classes, activation='softmax')
])
print(model.summary())
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
trainnn = model.fit(
    traindiv,
    epochs=3,
    validation_data=testdiv,
    validation_steps=testdiv.samples//testdiv.batch_size
)

test_loss, test_accuracy = model.evaluate(testdiv, steps=testdiv.samples)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
model.save("lungimagesets.h5")

#took 3 hrs to train and test, 95% accuracy 14% loss