import cv2
from time import sleep
import math
# Opens the Video file
cap = cv2.VideoCapture('/Users/beyrem/PycharmProjects/extract/loving.m4v')
i = 0
frameRate = cap.get(5)
while (cap.isOpened()):
    frameId = cap.get(1)
    ret, frame = cap.read()
    if ret == False:
        break
    if (frameId % 3 == 0):
        cv2.imwrite('/images/frame' + str(i) + '.jpg', frame)
        i += 1

cap.release()
cv2.destroyAllWindows()