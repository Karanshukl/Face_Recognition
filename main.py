from typing import Any, Sequence
import cv2
from numpy import dtype, floating, integer, ndarray

face_cap =  cv2.CascadeClassifier("C:/Users/Karan Shukla/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

vid_cap = cv2.VideoCapture(0)
while True:
    ret, vid_data = vid_cap.read()
    # Converts the color frame to grayscale for faster processing.
    col: cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = cv2.cvtColor(vid_data, cv2.COLOR_BGR2GRAY)
    face: Sequence[Sequence[int]] = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in face:
        cv2.rectangle(vid_data,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Karan Cam's", vid_data)

    if cv2.waitKey(10) == ord("s"):
        break

vid_cap.release()

