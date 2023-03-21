import keras
from keras import layers
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Set seed
tf.keras.utils.set_random_seed(0)

# Load image
img = cv2.imread("Car.png")
img = cv2.resize(img, (224, 224))

# VGG 16 Model
model = keras.Sequential()
# Block 1
# 1. convolution layer with 64 filters
model.add(layers.Conv2D(input_shape=(224, 224, 3), activation="relu", padding="same", filters=64, kernel_size=(3, 3)))
# 2. Convolution layer similar to the first one
model.add(layers.Conv2D(filters=64, padding="same", activation="relu", kernel_size=(3, 3)))
# 3. MaxPooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

# Block 2
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# Block 3
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# The further we go into the network, the features are extracted more in detail.

# Block 4
model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# The more we are going depth, the smaller the features are getting the more simpler, but the number of features is increasing

# Block 5
model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# Block 6
model.add(layers.Flatten())
model.add(layers.Dense(units=4096, activation="relu"))
model.add(layers.Dense(units=4096, activation="relu"))
model.add(layers.Dense(units=3, activation="softmax"))  # This model was trained to detect 1000 categories, it is always edited to the number of objects we want to detect for building our own model

# Build the model
model.build()
model.summary()

# result
# Predict the result of the image
feature_map = model.predict(np.array([img]))

# specify and understand the category of our image
print(feature_map)
# The output is a matrix of 3 numbers,

# for i in range(512):
#     feature_img = feature_map[0, :, :, i]
#     ax = plt.subplot(32, 16, i+1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.imshow(feature_img, cmap="gray")
# plt.show()
