import numpy as np
import cv2
import pickle
import time

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")


labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

print(labels)

profile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    faces_profile = profile_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45: #and conf <= 85:
            print(id_, conf)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 0
            cv2.putText(frame, name, (x, y), font, 0.5, color, stroke, cv2.LINE_AA)

        img_item = "faces/front_face.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0) #BGR
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x, y), (width, height), color, stroke)

    for (x, y, w, h) in faces_profile:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        img_item = "faces/profile-face.png"
        cv2.imwrite(img_item, roi_gray)

        color = (0, 255, 0) #BGR
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x, y), (width, height), color, stroke)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.2)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()