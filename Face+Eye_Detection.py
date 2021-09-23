import cv2, os

haarFace= "E:\\FaceDetection-And-More\\Haarcascade\\haarcascade_frontalface_default.xml"
haarEye = "E:\\FaceDetection-And-More\\Haarcascade\\haarcascade_eye.xml"

FaceDetect = cv2.CascadeClassifier(haarFace)
EyeDetect = cv2.CascadeClassifier(haarEye)

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    text = "No Face Detected"
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_feature = FaceDetect.detectMultiScale(grayImg, 1.3, 5)
    eye_feature = EyeDetect.detectMultiScale(grayImg, 1.3, 5)
    for (x,y,w,h) in face_feature:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    for (a,b) in eye_feature:
        cv2.circle(img, (a,b), 2, (0,0,255), 2)
        
