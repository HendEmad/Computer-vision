import keras
from keras import layers
import cv2

img = cv2.imread("Car.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (224, 224))  # Common size for classification neural network
height, width = img.shape

model = keras.Sequential()
# Parameters essential for creating the Conv2D layer are:
# input shape, filters, kernel_size
model.add(layers.Conv2D(input_shape=(height, width, 1), filters=64, kernel_size=(3, 3)))

model.summary()

# Access filters/parameters/weights/features
filters, _ = model.layers[0].get_weights() # The index is 0 for the 1st layer, 1 for the 2nd layer, and so on
print("Filters shape = ", filters.shape)  # (3, 3, 1, 64) => 64 images of size 3x3
# filters = cv2.resize(filters, (250, 250), interpolation=cv2.INTER_NEAREST)
# cv2.imshow("img", f)
# cv2.waitKey(0)

# Normalization step
print("F maximum and F minimum before normalization: ", filters.min(), ", ", filters.max())
# Normalize each filter (feature extraction image)
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
print("After normalization: ", filters.min(), ", ", filters.max())
# cv2.imshow("img", f)
# cv2.waitKey(0)

# Get the first 10 filters/ feature images
for i in range(10):
    f_10 = filters[:, :, :, i]
    f_10 = cv2.resize(f_10, (250, 250), interpolation=cv2.INTER_NEAREST)
    print(f_10)
    cv2.imshow(str(i), f_10)

cv2.waitKey(0)