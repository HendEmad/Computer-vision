# Import libraries
import keras
from keras.layers import Dense
import cv2

# Checking versions
# print(keras.__version__)
# print(cv2.__version__)

# Load image using OpenCV in RGB
img = cv2.imread("Car.png")
cv2.imshow("Original car image", img)
# cv2.waitKey(0)  # To keep the image on hold

# Image info
# print(img.shape)  # Size
# print(img)  # List

# Read the image in gray scale
grayImg = cv2.imread("Car.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Car image in gray scale", grayImg)
# cv2.waitKey(0)
# print(grayImg.shape)
# print(grayImg)

# Passing the input layer into the neural network
rows, columns = grayImg.shape  # Image height and width

# Neural network structure
input_layer = keras.Input(shape=(rows, columns))
layer_1 = Dense(64)(input_layer)
layer_2 = Dense(32)(layer_1)
output = Dense(2)(layer_2)

# Define the model
model = keras.Model(inputs=input_layer, outputs=output)

# Model summary to make sure that everything is on
model.summary()