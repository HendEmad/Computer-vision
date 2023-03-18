import keras
from keras import layers
import cv2
import numpy as np

# Load and Pre-process image
img = cv2.imread("Car.png", )
height, width, channels = img.shape
print("Height = {}, width = {}, channels = {}".format(height, width, channels))

# Create a sequential API model
model = keras.Sequential()
model.add(layers.Input(shape=(height, width, channels)))
model.add(layers.Dense(32))
model.add(layers.Dense(16))  # Adding a new layer on the top of the first layer and connect between them
model.add(layers.Dense(2))

# Change the size of image if needed
preprocessed_img = np.array([img])

# Passing our image into the model
result = model(preprocessed_img)
print(result)