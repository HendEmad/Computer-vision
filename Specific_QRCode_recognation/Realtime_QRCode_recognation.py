# Define the function for real-time testing
import cv2
import keras


def classify_webcam():
    # Load the trained model
    model = keras.models.load_model('my_model.h5')

    # Open the camera
    cap = cv2.VideoCapture(1)

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Preprocess the image
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = img.reshape(1, 224, 224, 3)

        # Run the classification model
        prediction = model.predict(img)[0][0]
        if prediction > 0.5:
            label = 'not_cardiac_arrest_patient'
        else:
            label = 'cardiac_arrest_patient'

        # Display the result
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        # Wait for a key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()


# Call the real-time testing function
classify_webcam()