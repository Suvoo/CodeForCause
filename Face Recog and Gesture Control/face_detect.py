import cv2

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("C:\\Users\\91865\\Desktop\\CodeForCause\\Face Recog and Gesture Control\\haarcascade_frontalface_default.xml")

while True:

    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)

        for face in faces:
            x, y, w, h = face

            cut = frame[y:y + h, x:x + w]

            fix = cv2.resize(cut, (100, 100)) #has 100 square features 100 x 100
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY) # to convert to gray

        cv2.imshow("My Screen", frame)
        cv2.imshow("My Face", gray)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
