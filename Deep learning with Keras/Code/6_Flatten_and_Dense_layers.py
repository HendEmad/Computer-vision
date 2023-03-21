import keras
import cv2
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load image
img_size = 32
img = cv2.imread("Car.png")
img = cv2.resize(img, (img_size, img_size))


# CNN Model
model = keras.Sequential()
model.add(layers.Conv2D(input_shape=(img_size, img_size, 3), filters=64, kernel_size=(3, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(units=10))

result = model.predict(np.array([img]))
print("Extracted features after convolution and pooling layers: \n", result)
print("The features shape: ", result.shape)
# The result shape after convolution and pooling = (1, 29, 29, 64)
# The result shape after performing flatten = (1, 53824) ==> (1, 29*29*64)
# The result shape after performing dense = (1, 10)