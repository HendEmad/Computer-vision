import cv2
import numpy as np

def classier_image(img):
    # 1) simplify the image by dividing it by 255
    img = img / 255
    # print(img)

    # 2) Flatten the image
    img_flattened = img.flatten()
    # print(img_flattened)

    # 3) Filter
    filter = [1, -1, 1, 1, -1, 1, 1, -1, 1]

    # 4) Multiply filter * flattened image
    convolution = img_flattened * filter
    # print(convolution)

    # 5) Sum of the convolution
    sum_convolution = sum(convolution)
    # print(sum_convolution)

    # 6) Classification condition
    if sum_convolution== 1:
        return "Horizontal"
    else:
        return "Vertical"


img = cv2.imread("images/vertical.png", cv2.IMREAD_GRAYSCALE)
result = classier_image(img)  # vertical
print(result)

img2 = cv2.imread("images/horizontal.png", cv2.IMREAD_GRAYSCALE)
result2 = classier_image(img2)  # Horizontal
print(result2)

img3 = cv2.imread("images/vertical2.png", cv2.IMREAD_GRAYSCALE)
result3 = classier_image(img3)  # Vertical
print(result3)

img4 = cv2.imread("images/horizontal2.png", cv2.IMREAD_GRAYSCALE)
result4 = classier_image(img4)  # Horizontal
print(result4)
