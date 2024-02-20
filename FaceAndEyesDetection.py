
import cv2
import numpy as np
import time

# Load pre-trained face detection model
face_model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_prototxt_path = "deploy.prototxt.txt"
face_net = cv2.dnn.readNetFromCaffe(face_prototxt_path, face_model_path)

# Load Haar Cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

start_time = time.time()
look_away_time = 0
look_away_threshold = 20  # in seconds
look_away_alert_threshold = 20 * 60  # 20 minutes in seconds
print("program running")

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    # Pre-process the frame for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold for face detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            # Extract face region
            face_roi = frame[y:y2, x:x2]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Perform eye detection using Haar Cascade
            eyes = eye_cascade.detectMultiScale(gray_face)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    cv2.imshow("Face and Eyes", frame)

    if time.time() - start_time >= look_away_alert_threshold:
        print("Look away! You've been staring at the screen for more than 20 minutes.")
        start_time = time.time()  # Reset the timer after alert

    if cv2.waitKey(1) == 13:
        break

    if len(detections) == 0:
        look_away_time = time.time()  # Update look away time if the user looks away

cap.release()
cv2.destroyAllWindows()
