import cv2
import numpy as np

cap = cv2.VideoCapture(0)
back = cv2.imread('./image.jpg')
kernel = np.ones((5,5), np.uint8)  #to start morphology

while cap.isOpened():
    ret, frame = cap.read() #read frame from camera

    if ret:
        # HSV : Hue, Saturation, value
        # color, amount with white, amount with black
        # HSV  like human eyes, but RGB just mixture

        # open CV reads BGR, then to convert to HSV

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv",hsv)
        # to get hsv value
        # lower : hue-10,100,100  higher: h+10,255 255

        blue = np.uint8([[[255,0,0]]]) #bgr of blue
        hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
        # print(hsv_blue)  #to get hsv of blue color which will be used

        # threshold hsv to only get blue
        lower_blue = np.array([0,100,100])
        upper_blue = np.array(([130,255,255]))

        # to make blue range disappear
        mask = cv2.inRange(hsv,lower_blue,upper_blue) #ignore all colors apart from mask
        # cv2.imshow("mask", mask)

        # all things blue
        part1 = cv2.bitwise_and(back, back, mask=mask) #masking all blue with background image
        # cv2.imshow("part1", part1)

        mask = cv2.bitwise_not(mask)

        # part 2 is all things not blue
        part2 = cv2.bitwise_and(frame, frame, mask=mask)
        #cv2.imshow("mask", part2)

        # we want part1+part2
        cv2.imshow("cloak", part1 + part2)
        # done, but to make images very clear use morphology

        img_erosion = cv2.erode(back, kernel, iterations=1)
        img_dilation = cv2.dilate(back, kernel, iterations=1)

        # cv2.imshow('Input', back)

        # cv2.imshow('Dilation', img_dilation)
        # cv2.imshow('Erosion', img_erosion)


        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()