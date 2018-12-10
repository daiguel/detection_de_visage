import cv2
import numpy as np
#initialisation du calassifier
faceDetect=cv2.CascadeClassifier('data\haarcascades\haarcascade_frontalface_default.xml')

cam=cv2.VideoCapture(0)

while (True):
    ret,img=cam.read()

    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces=faceDetect.detectMultiScale(img,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),1)

    cv2.imshow("firstLive", img)
    cv2.waitKey(0)

cv2.destroyWindow("firstLive")
cam.release()