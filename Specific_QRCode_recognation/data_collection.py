import cv2

cap = cv2.VideoCapture(1)
file_path = 'C:\\Users\\Data\\PycharmProjects\\Embedded_AI_Quadcopter\\data'
file_name = 'cardiac_arrest_patient'

img_width = 256
img_height = 256
num_images = 100

for i in range(num_images):
    ret, frame = cap.read()

    resized_frame = cv2.resize(frame, (img_width, img_height))
    cv2.imshow('Webcam data', resized_frame)
    cv2.waitKey(1000)

    img_path = file_path + file_name + str(i) + '.jpg'
    cv2.imwrite(img_path, resized_frame)

cap.release()
cv2.destroyAllWindows()