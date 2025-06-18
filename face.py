import cv2
from djitellopy import *

tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

faceClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect():
    frame = frame_read.frame
    if len(frame.shape) == 2:  # Grayscale image
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:  # RGBA image
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassifier.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return faces

while True:
    faces = detect()

    cv2.imshow("Face Detection", frame_read.frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
