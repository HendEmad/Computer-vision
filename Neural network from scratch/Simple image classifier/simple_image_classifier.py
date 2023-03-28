import cv2
import numpy as np

# load images
vertical = cv2.imread("images/vertical.png", cv2.IMREAD_GRAYSCALE)
horizontal = cv2.imread("images/horizontal.png", cv2.IMREAD_GRAYSCALE)

# Printing the images arrays
# print("Vertical image: \n", vertical, "\n")
# print("horizontal image: \n", horizontal)


# image preparation
# 1) Simplify the images (Image normalization)
vertical = vertical / 255
horizontal = horizontal / 255
# print("Vertical image: \n", vertical, "\n")
# print("horizontal image: \n", horizontal)

# 2) flatten the images
vertical_flattened = vertical.flatten()
horizontal_flattened = horizontal.flatten()
# print("Vertical image: ", vertical_flattened)
# print("Horizontal image: ", horizontal_flattened)


# Create image recognition classifier
# horizontal_sum = sum(horizontal_flattened)
# print("horizontal sum = ", horizontal_sum)  # 3.0
# vertical_sum = sum(vertical_flattened)
# print("vertical sum = ", vertical_sum)  # 3.0

# 1) create the filter
# filter_no = [1, -1, 1, -1, 1, -1, 1, -1, 1]
# print(horizontal_flattened * filter)
# print(sum(horizontal_flattened*filter))  # -1.0
# print(vertical_flattened * filter)
# print(sum(vertical_flattened*filter))  # -1.0

filter = [1, -1, 1, 1, -1, 1, 1, -1, 1]
sum = sum(horizontal_flattened * filter)  # Horizontal
# sum = sum(vertical_flattened * filter)  # Vertical

if sum == 1:
    print("Horizontal")
else:
    print("Vertical")

# cv2.imshow("vertical image", cv2.resize(vertical, (500, 500), interpolation=0))
# cv2.imshow("horizontal image", cv2.resize(horizontal, (500, 500), interpolation=0))
# cv2.imshow("vertical flattened", cv2.resize(vertical_flattened, (100, 900), interpolation=0))
# cv2.imshow("Horizontal flattened", cv2.resize(horizontal_flattened, (100, 450), interpolation=0))
# cv2.waitKey(0)
