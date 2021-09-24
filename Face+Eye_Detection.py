import cv2

# Loading the models for face & eye detection
haarFace= "Haarcascade\haarcascade_frontalface_default.xml"
haarRightEye = "Haarcascade\haarcascade_righteye_2splits.xml"
haarleftEye = "Haarcascade\haarcascade_lefteye_2splits.xml"

# Initializing models using CascadeClassifier 
# This is a smart solution (applying limited feature in stages)
# used to find only the possible pixels containing face or eyes
# and then applying remaining features in stages to that selected area

FaceDetect = cv2.CascadeClassifier(haarFace)
R_EyeDetect = cv2.CascadeClassifier(haarRightEye)
L_EyeDetect = cv2.CascadeClassifier(haarleftEye)

# using the webcam hence VideoCapture(0)
# if you are using external camera, then you should try VideoCapture(1) or VideoCapture(2)
cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    facetext = "No Face Detected"
    eyetext = "No Eyes Detected" 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #grayImg = cv2.resize(grayImg, (21,21), fx=1, fy=1)
    # Finding out coordinates of the feature we want to pin-point
    face_feature = FaceDetect.detectMultiScale(grayImg, 1.3, 4)
    R_eye_feature = R_EyeDetect.detectMultiScale(grayImg)
    L_eye_feature = L_EyeDetect.detectMultiScale(grayImg)

    # Drawing and adding text simultaneously

    # For face
    for (x,y,w,h) in face_feature:
        facetext = "Face Detected"
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    # For left eye
    for (a,b,c,d) in L_eye_feature:
        eyetext = "Eyes Detected"
        cv2.rectangle(img, (a,b), (a+c,b+d), (0,255,255), 2)
        #cv2.circle(img, (a,b), 20, (0,0,255), 2)
        cv2.putText(img, "Left-Eye", (a-3,b-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,255))

    # For right eye 
    for (p,q,r,s) in R_eye_feature:
        eyetext = "Eyes Detected"
        cv2.rectangle(img, (p,q), (p+r,q+s), (0,255,255), 2)
        #cv2.circle(img, (p,q), 20, (0,0,255), 2)
        cv2.putText(img, "Right-Eye", (p-3,q-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,255))

    # printing text on terminal
    print(facetext)
    print(eyetext)

    # Adding text on the screen
    cv2.putText(img, facetext, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,255,0), thickness=2)
    cv2.putText(img, eyetext, (480,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,255), thickness=2)

    # Output
    cv2.imshow('Face+Eye_Detection', img)

    # Coming out of infinite loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()