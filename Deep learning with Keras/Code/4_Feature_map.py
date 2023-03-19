import keras
import cv2
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("Car.png")
img = cv2.resize(img, (224, 224))
height, width, channels = img.shape
# cv2.imshow("img", img)
# cv2.waitKey(0)

# Model
model = keras.Sequential()
model.add(layers.Conv2D(input_shape=(height, width, channels), filters=64, kernel_size=(3, 3)))

model.summary()

# pass the image into the model
feature_map = model.predict(np.array([img]))
# print(feature_map.shape)  # 64 feature images of size 222x222 for one input image
# print(feature_map)  # The output is the feature map

# Visualize the feature map (64 images)
#  Extract only onr
feature_img = feature_map[0, :, :, 0]
plt.imshow(feature_img, cmap="gray")  # To display the image in grayscale
plt.show()

# Extract all images
for i in range(64):
    feature_img = feature_map[0, :, :, i]
    ax = plt.subplot(8, 8, i+1)  # Put number of image, foe example: for cell 0, column 0 => image 1
    # For `subplot` function, we need to pass 3 parameter:
    # no.of vertical images, no.of horizontal images, index of each single image
    # Remove the ticks (0 -> 222) that are around each image
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_img, cmap="gray")
plt.show()