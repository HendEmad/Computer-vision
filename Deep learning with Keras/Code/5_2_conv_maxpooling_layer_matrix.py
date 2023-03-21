import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras import layers

# Set random seed
# tf.random.set_seed(0)
tf.keras.utils.set_random_seed(0)

# Load image
img = cv2.imread("Car.png")
img = cv2.resize(img, (7, 7))

# CNN Model
model = keras.Sequential()
model.add(layers.Conv2D(input_shape=(7, 7, 3), filters=64, kernel_size=(3, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
feature_map = model.predict(np.array([img]))

feature_img = feature_map[0, :, :, 0]
print("Image pixels after convolution and pooling\n", feature_img)
plt.figure("Image after convolution and pooling")
plt.imshow(feature_img, cmap="gray")
plt.show()

