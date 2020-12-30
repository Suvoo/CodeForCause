import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.load("C:\\Users\\91865\\Desktop\\CodeForCause\\Face Recog and Gesture Control\\face_data.npy")

print(data.shape, data.dtype)  # no.of entries, features 100001

X = data[:, 1:].astype(int)  # to convert to int, not unicode
y = data[:, 0]

model = KNeighborsClassifier()
model.fit(X, y)

cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("C:\\Users\\91865\\Desktop\\CodeForCause\\Face Recog and Gesture Control\\haarcascade_frontalface_default.xml")
while True:

    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)

        for face in faces:
            x, y, w, h = face

            cut = frame[y:y + h, x:x + w]

            fix = cv2.resize(cut, (100, 100))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

            out = model.predict([gray.flatten()])  # has the correct name of figure

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # color, thickness of rectangle on face

            cv2.putText(frame, str(out[0]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2) #  to put name

            print(out)

            cv2.imshow("My Face", gray)

        cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
