import cv2, os

haarfile = "E:\FaceDetection-And-More\Haarcascade\haarcascade_eye.xml"

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    img = cv2.resize(img, dsize= 0, fx = 0.5, fy = 0.5)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    gaussianBlur = cv2.GaussianBlur(grayImg, (21,21), 0, 2)
    x , y, w, h = cv2.CascadeClassifier(haarfile)
    cv2.boundingRect(img, (x, y, w, h), color = (0,255, 0), 1)
