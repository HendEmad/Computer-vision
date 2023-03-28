import cv2
import numpy as np

height, width = 3, 3
# 1) vertical.png
vertical = np.zeros((height, width), np.uint8)
vertical[:, 0] = (0, 0, 0)
vertical[:, 1] = (255, 255, 255)
vertical[:, 2] = (0, 0, 0)

# 2) horizontal.png
horizontal = np.zeros((height, width), np.uint8)
horizontal[0, :] = (0, 0, 0)
horizontal[1, :] = (255, 255, 255)
horizontal[2, :] = (0, 0, 0)

# 3) vertical2.png
vertical2 = np.zeros((height, width), np.uint8)
vertical2[:, 0] = (255, 255, 255)
vertical2[:, 1] = (0, 0, 0)
vertical2[:, 2] = (0, 0, 0)

# 4) horizontal2.png
horizontal2 = np.zeros((height, width), np.uint8)
horizontal2[0, :] = (0, 0, 0)
horizontal2[1, :] = (0, 0, 0)
horizontal2[2, :] = (255, 255, 255)

# Save images
vertical_img = cv2.imwrite(r'images/vertical.png', vertical)
horizontal_img = cv2.imwrite(r'images/horizontal.png', horizontal)
vertical_img2 = cv2.imwrite(r'images/vertical2.png', vertical2)
horizontal_img2 = cv2.imwrite(r'images/horizontal2.png', horizontal2)
