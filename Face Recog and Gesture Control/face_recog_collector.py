import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("C:\\Users\\91865\\Desktop\\CodeForCause\\Face Recog and Gesture Control\\haarcascade_frontalface_default.xml")

name = input("Enter your name : ")

frames = []
outputs = []  # for saving all file names. instead for one, use cv.imwrite

while True:

    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)

        for face in faces:
            x, y, w, h = face

            cut = frame[y:y+h, x:x+w]

            fix = cv2.resize(cut, (100, 100))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

        cv2.imshow("My Screen", frame)
        cv2.imshow("My Face", gray)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    if key == ord("c"):
        # cv2.imwrite(name + ".jpg", frame)
        frames.append(gray.flatten())  # to flatten the image to show all features --> [.. , .. , .. ,]
        outputs.append([name])  # to store the names

X = np.array(frames)  # for the model, convert ot numpy
y = np.array(outputs)  # same for y

data = np.hstack([y, X])  # to store in horizontal stack like 0th col is y then rest is X
# print(data.shape) -->  this shows (n.of times c pressed, 100001)

f_name = "face_data.npy"  # to store for permanent memory
#  to store data, we need to vertically stack over older items

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])

np.save(f_name, data)

cap.release()
cv2.destroyAllWindows()