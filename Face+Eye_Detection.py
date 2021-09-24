import cv2

haarFace= "Haarcascade\haarcascade_frontalface_default.xml"
haarRightEye = "Haarcascade\haarcascade_righteye_2splits.xml"
haarleftEye = "Haarcascade\haarcascade_lefteye_2splits.xml"

FaceDetect = cv2.CascadeClassifier(haarFace)
R_EyeDetect = cv2.CascadeClassifier(haarRightEye)
L_EyeDetect = cv2.CascadeClassifier(haarleftEye)

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    facetext = "No Face Detected"
    eyetext = "No Eyes Detected" 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #grayImg = cv2.resize(grayImg, (21,21), fx=1, fy=1)
    face_feature = FaceDetect.detectMultiScale(grayImg, 1.3, 4)
    R_eye_feature = R_EyeDetect.detectMultiScale(grayImg)
    L_eye_feature = L_EyeDetect.detectMultiScale(grayImg)
    for (x,y,w,h) in face_feature:
        facetext = "Face Detected"
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    for (a,b,c,d) in L_eye_feature:
        eyetext = "Eyes Detected"
        cv2.rectangle(img, (a,b), (a+c,b+d), (0,255,0), 2)
        #cv2.circle(img, (a,b), 20, (0,0,255), 2)
        cv2.putText(img, "Left-Eye", (a-3,b-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255,0,0))
    for (p,q,r,s) in R_eye_feature:
        eyetext = "Eyes Detected"
        cv2.rectangle(img, (p,q), (p+r,q+s), (0,255,0), 2)
        #cv2.circle(img, (p,q), 20, (0,0,255), 2)
        cv2.putText(img, "Right-Eye", (p-3,q-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255,0,0))
    print(facetext)
    print(eyetext)
    cv2.putText(img, facetext, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,255,0), thickness=2)
    cv2.putText(img, eyetext, (480,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,255), thickness=2)
    cv2.imshow('Face+Eye_Detection', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()